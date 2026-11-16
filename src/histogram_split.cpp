#include <Rcpp.h>
#include <vector>
#include <chrono>
using namespace Rcpp;

// The optimized kernel does not call ranger or wrap a ready-made tree library.
// Column-major traversal matches R matrices. No n-by-K temporary is created.
// [[Rcpp::export]]
NumericVector split_rcpp(IntegerMatrix x, IntegerVector y, int bins=32, int min_leaf=5) {
    const int n=x.nrow(), p=x.ncol();
    if (n < 1 || p < 1 || y.size()!=n || bins<2 || min_leaf<1)
        stop("Invalid dimensions, bins or leaf size");
    double positive=0;
    for (int i=0;i<n;++i) {
        if (y[i]!=0 && y[i]!=1) stop("Labels must be binary");
        positive+=y[i];
    }
    double best=0;
    int feature=-1, threshold=-1;
    std::vector<double> counts(bins), positives(bins);
    for (int j=0;j<p;++j) {
        std::fill(counts.begin(), counts.end(), 0);
        std::fill(positives.begin(), positives.end(), 0);
        for (int i=0;i<n;++i) {
            const int b=x(i,j);
            if (b<0 || b>=bins) stop("Bin out of range");
            counts[b]+=1;
            positives[b]+=y[i];
        }
        double nl=0, pl=0;
        for (int b=0;b<bins-1;++b) {
            nl+=counts[b]; pl+=positives[b];
            const double nr=n-nl, pr=positive-pl;
            if (nl<min_leaf || nr<min_leaf) continue;
            const double gain=2*positive*(n-positive)/(double(n)*n)
                -(2*pl*(nl-pl)/nl+2*pr*(nr-pr)/nr)/n;
            if (gain>best+1e-14) { best=gain; feature=j; threshold=b; }
        }
    }
    return NumericVector::create(feature, threshold, best);
}

// [[Rcpp::export]]
double monotonic_seconds() {
    // R's elapsed clock is coarse on some Windows builds; use C++ steady_clock.
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
