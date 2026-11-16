dir.create(".Rlib", showWarnings = FALSE)
# Match the worker environment: project library plus system/site libraries.
# Do not accidentally reuse a package found only in a different user's library.
.libPaths(c(normalizePath(".Rlib"), .Library.site, .Library))
packages <- c("Rcpp", "ranger", "jsonlite")
missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) install.packages(missing, repos = "https://cloud.r-project.org", lib = ".Rlib")
stopifnot(all(vapply(packages, requireNamespace, logical(1), quietly = TRUE)))
cat("R dependencies ready in project-local .Rlib\n")
print(sessionInfo())
