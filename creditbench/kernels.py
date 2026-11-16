"""Original binary-Gini histogram split search, with four language counterparts.

Bins are 0..K-1. Return (0-based feature, 0-based split bin, Gini gain).
Tie rule: first feature/threshold, gains within 1e-14 treated as tied.
"""
import numpy as np

TIE_TOL = 1e-14


def validate(x, y, bins, min_leaf):
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim != 2 or y.ndim != 1 or len(y) != len(x) or not x.shape[0] or not x.shape[1]:
        raise ValueError("Require nonempty 2D x and matching 1D y")
    if not np.issubdtype(x.dtype, np.integer) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("x and y must have integer dtype")
    if bins < 2 or min_leaf < 1 or x.min() < 0 or x.max() >= bins or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid bins, labels or minimum leaf size")


def _gain(n, positive, left_n, left_positive):
    right_n = n-left_n
    right_positive = positive-left_positive
    return (2.0*positive*(n-positive)/n**2
            - (2.0*left_positive*(left_n-left_positive)/left_n
               + 2.0*right_positive*(right_n-right_positive)/right_n)/n)


def split_python(x, y, bins=32, min_leaf=5):
    """Pure Python reference: histogram construction and scan, O(n*p + K*p)."""
    validate(x, y, bins, min_leaf)
    n, p = x.shape
    total = int(sum(int(v) for v in y))
    best = (-1, -1, 0.0)
    for j in range(p):
        counts, positives = [0]*bins, [0]*bins
        for i in range(n):
            b = int(x[i, j])
            counts[b] += 1
            positives[b] += int(y[i])
        nl = pl = 0
        for b in range(bins-1):
            nl += counts[b]
            pl += positives[b]
            if nl < min_leaf or n-nl < min_leaf:
                continue
            gain = _gain(n, total, nl, pl)
            if gain > best[2] + TIE_TOL:
                best = (j, b, float(gain))
    return best


def split_numpy(x, y, bins=32, min_leaf=5):
    """Same algorithm, NumPy histogram/cumulative-sum primitives."""
    validate(x, y, bins, min_leaf)
    n, p = x.shape
    total = int(y.sum())
    best = (-1, -1, 0.0)
    for j in range(p):
        nl = np.cumsum(np.bincount(x[:, j], minlength=bins))[:-1]
        pl = np.cumsum(np.bincount(x[:, j], weights=y, minlength=bins))[:-1]
        valid = np.flatnonzero((nl >= min_leaf) & (n-nl >= min_leaf))
        for b, gain in zip(valid, _gain(n, total, nl[valid], pl[valid])):
            if gain > best[2] + TIE_TOL:
                best = (j, int(b), float(gain))
    return best


def split_bruteforce(x, y, bins=32, min_leaf=5):
    """Independent O(n*p*K) reference; deliberately outside benchmark timings."""
    validate(x, y, bins, min_leaf)
    best = (-1, -1, 0.0)
    n = len(y)
    parent = 2*y.mean()*(1-y.mean())
    for j in range(x.shape[1]):
        for b in range(bins-1):
            mask = x[:, j] <= b
            left, right = y[mask], y[~mask]
            if min(len(left), len(right)) < min_leaf:
                continue
            child = sum(len(z)/n * 2*z.mean()*(1-z.mean()) for z in (left, right))
            gain = float(parent-child)
            if gain > best[2] + TIE_TOL:
                best = (j, b, gain)
    return best


class QuantileBinner:
    def __init__(self, bins=32):
        self.bins = bins

    def fit(self, x):
        self.edges = [np.unique(np.quantile(x[:, j], np.arange(1, self.bins)/self.bins))
                      for j in range(x.shape[1])]
        return self

    def transform(self, x):
        return np.column_stack([np.searchsorted(e, x[:, j], side="left") for j, e in enumerate(self.edges)]).astype(np.int32)


class HistogramForest:
    """Small educational forest using our split kernel; not a production substitute.

    Quantile bins are an approximation to exact CART candidate thresholds.
    Tree bootstrap + sqrt(p) features per node; leaf probabilities are averaged.
    """
    def __init__(self, trees=20, max_depth=8, min_leaf=5, bins=32, seed=42, kernel=split_numpy):
        self.trees, self.max_depth, self.min_leaf = trees, max_depth, min_leaf
        self.bins, self.seed, self.kernel = bins, seed, kernel

    def fit(self, x, y):
        self.binner = QuantileBinner(self.bins).fit(x)
        z = self.binner.transform(x)
        rng = np.random.default_rng(self.seed)
        mtry = max(1, int(np.sqrt(z.shape[1])))

        def grow(ids, depth):
            probability = float(y[ids].mean())
            if depth >= self.max_depth or len(ids) < 2*self.min_leaf or probability in (0, 1):
                return probability
            features = np.sort(rng.choice(z.shape[1], mtry, replace=False))
            j, cut, gain = self.kernel(z[ids][:, features], y[ids], self.bins, self.min_leaf)
            if j < 0:
                return probability
            feature = int(features[j])
            left = z[ids, feature] <= cut
            return (feature, cut, grow(ids[left], depth+1), grow(ids[~left], depth+1))

        self.forest = [grow(rng.integers(len(y), size=len(y)), 0) for _ in range(self.trees)]
        return self

    def predict_proba(self, x):
        z = self.binner.transform(x)

        def predict(node, row):
            while isinstance(node, tuple):
                feature, cut, left, right = node
                node = left if row[feature] <= cut else right
            return node

        return np.mean([[predict(tree, row) for row in z] for tree in self.forest], axis=0)
