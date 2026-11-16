library(jsonlite)
source("R/kernels.R")
source("R/compile.R") # Compilation and loading are excluded from kernel timers.
args <- commandArgs(trailingOnly=TRUE)
spec <- fromJSON(args[1])
records <- list()
for (i in seq_len(nrow(spec$cases))) {
  case <- spec$cases[i,]
  x <- as.matrix(read.csv(case$x, header=FALSE)); storage.mode(x) <- "integer"
  y <- scan(case$y, quiet=TRUE, what=integer())
  for (engine in c("R", "Rcpp")) {
    fun <- if(engine=="R") split_r else split_rcpp
    actual <- fun(x, y, spec$bins, spec$min_leaf)
    expected <- c(case$feature, case$threshold, case$gain)
    stopifnot(all(actual[1:2] == expected[1:2]), max(abs(actual-expected)) <= 1e-12)
    # Calibrate an untimed batch so even sub-millisecond kernels have measurable duration.
    batch <- 1L
    repeat {
      start <- monotonic_seconds()
      for (k in seq_len(batch)) invisible(fun(x, y, spec$bins, spec$min_leaf))
      elapsed <- monotonic_seconds()-start
      if (elapsed >= .03 || batch >= 16384L) break
      batch <- batch*2L
    }
    for (r in seq_len(spec$repeats)) {
      gc()
      start <- monotonic_seconds()
      for (k in seq_len(batch)) invisible(fun(x, y, spec$bins, spec$min_leaf))
      elapsed <- monotonic_seconds()-start
      records[[length(records)+1L]] <- data.frame(engine=engine, n=nrow(x), p=ncol(x), bins=spec$bins,
        repeat_id=r-1L, batch=batch, seconds=elapsed/batch,
        feature=actual[1], threshold=actual[2], gain=actual[3])
    }
  }
}
write.csv(do.call(rbind, records), args[2], row.names=FALSE)
capture.output(sessionInfo(), file=file.path(dirname(args[2]), "R_sessionInfo.txt"))
