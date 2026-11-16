dir.create("build/rcpp", recursive=TRUE, showWarnings=FALSE)
Rcpp::sourceCpp("src/histogram_split.cpp", cacheDir="build/rcpp", rebuild=FALSE, verbose=FALSE)
