library(jsonlite)
source("R/kernels.R")
source("R/compile.R")
cases <- fromJSON(commandArgs(trailingOnly=TRUE)[1], simplifyVector=FALSE)
for (case in cases) {
  x <- as.matrix(read.csv(case$x, header=FALSE))
  storage.mode(x) <- "integer"
  y <- scan(case$y, quiet=TRUE, what=integer())
  expected <- unlist(case$expected)
  for (fun in list(split_r, split_rcpp)) {
    actual <- fun(x, y, case$bins, case$min_leaf)
    stopifnot(identical(as.integer(actual[1:2]), as.integer(expected[1:2])),
              max(abs(actual-expected)) <= 1e-12)
  }
}
bad <- matrix(c(0L, 1L, NA_integer_, 2L), ncol=1L)
stopifnot(inherits(try(split_rcpp(bad, c(0L,1L,0L,1L), 4L, 1L), silent=TRUE), "try-error"))
cat("R and Rcpp matched independent reference for", length(cases), "fixtures.\n")
