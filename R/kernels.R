# Same O(n*p + K*p) histogram algorithm as Python/Rcpp; 0-based bins and result.
split_r <- function(x, y, bins = 32L, min_leaf = 5L) {
  stopifnot(is.matrix(x), nrow(x) > 0L, ncol(x) > 0L, length(y) == nrow(x),
            all(is.finite(x)), all(is.finite(y)), all(x == floor(x)),
            all(y %in% c(0L, 1L)), all(x >= 0L & x < bins), bins >= 2L, min_leaf >= 1L)
  n <- nrow(x)
  positive <- sum(y)
  best <- c(-1, -1, 0)
  for (j in seq_len(ncol(x))) {
    counts <- positives <- numeric(bins)
    for (i in seq_len(n)) {
      b <- x[i,j] + 1L
      counts[b] <- counts[b] + 1
      positives[b] <- positives[b] + y[i]
    }
    nl <- pl <- 0
    for (b in seq_len(bins-1L)) {
      nl <- nl + counts[b]
      pl <- pl + positives[b]
      nr <- n-nl
      if (nl < min_leaf || nr < min_leaf) next
      pr <- positive-pl
      gain <- 2*positive*(n-positive)/n^2 - (2*pl*(nl-pl)/nl + 2*pr*(nr-pr)/nr)/n
      if (gain > best[3] + 1e-14) best <- c(j-1L, b-1L, gain)
    }
  }
  best
}
