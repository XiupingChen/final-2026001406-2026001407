args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript R/worker.R SPEC.json")
library(jsonlite)
library(ranger)
# This worker uses ranger only; Rcpp is compiled separately for the kernel experiment.
monotonic_seconds <- function() as.numeric(proc.time()[["elapsed"]])
s <- fromJSON(args[1])
if (!"min.bucket" %in% names(formals(ranger::ranger))) stop("Install ranger >= 0.17.0 (min.bucket required)")
load_state <- function(s) {
  data <- read.csv(file.path(s$prepared, "dataset.csv"), check.names = FALSE)
  ids <- scan(file.path(s$scenario_dir, "train.csv"), quiet = TRUE) + 1L
  validation <- scan(file.path(s$prepared, "validation.csv"), quiet = TRUE) + 1L
  test <- scan(file.path(s$prepared, "test.csv"), quiet = TRUE) + 1L
  list(x = data[ids, -ncol(data), drop = FALSE], y = factor(data$target[ids], levels=c(0,1)),
       eval_x = data[c(validation,test), -ncol(data), drop = FALSE],
       boots = as.matrix(read.csv(file.path(s$scenario_dir, "bootstrap.csv"), header=FALSE)) + 1L,
       seeds = scan(file.path(s$scenario_dir, "seeds.csv"), quiet=TRUE))
}
task <- function(b, warm = FALSE) {
  start <- monotonic_seconds()
  ids <- state$boots[b,]
  fit_start <- monotonic_seconds()
  model <- ranger::ranger(x=state$x[ids,,drop=FALSE], y=state$y[ids],
                         probability=TRUE, num.trees=if(warm) min(3L,s$trees) else s$trees,
                         mtry=s$mtry, min.node.size=2L*s$min_leaf, min.bucket=s$min_leaf,
                         max.depth=s$max_depth, splitrule="gini", replace=TRUE, sample.fraction=1,
                         num.threads=s$inner, seed=state$seeds[b], verbose=FALSE)
  fit_seconds <- monotonic_seconds()-fit_start
  predict_start <- monotonic_seconds()
  prediction <- predict(model, data=state$eval_x, num.threads=s$inner)$predictions[,"1"]
  predict_seconds <- monotonic_seconds()-predict_start
  list(prediction=prediction, seconds=monotonic_seconds()-start,
       fit_seconds=fit_seconds, predict_seconds=predict_seconds)
}
start <- monotonic_seconds()
cluster <- NULL
if (s$outer == 1L) {
  state <- load_state(s)
  invisible(task(1L, warm=TRUE))
} else {
  # PSOCK works on Windows, macOS and Linux; no fork-only mclapply.
  cluster <- parallel::makePSOCKcluster(s$outer, outfile="", setup_timeout=60)
  parallel::clusterExport(cluster, c("s", "load_state", "task", "monotonic_seconds"), envir=.GlobalEnv)
  parallel::clusterEvalQ(cluster, {
    library(ranger)
    state <- load_state(s)
    invisible(task(1L, warm=TRUE))
    NULL
  })
}
initialization_seconds <- monotonic_seconds()-start
start <- monotonic_seconds()
measured <- tryCatch({
  values <- if (is.null(cluster)) lapply(seq_len(s$bootstrap), task)
            else parallel::parLapply(cluster, seq_len(s$bootstrap), task)
  list(values=values, seconds=monotonic_seconds()-start)
}, finally={if (!is.null(cluster)) parallel::stopCluster(cluster)})
values <- measured$values
prediction <- do.call(cbind, lapply(values, `[[`, "prediction"))
write.table(prediction, s$prediction_path, sep=",", row.names=FALSE, col.names=FALSE, quote=FALSE)
write_json(list(core_seconds=measured$seconds, initialization_seconds=initialization_seconds,
                task_seconds=vapply(values, `[[`, numeric(1), "seconds"),
                fit_seconds=vapply(values, `[[`, numeric(1), "fit_seconds"),
                predict_seconds=vapply(values, `[[`, numeric(1), "predict_seconds")),
           s$result_path, auto_unbox=TRUE, digits=16)
