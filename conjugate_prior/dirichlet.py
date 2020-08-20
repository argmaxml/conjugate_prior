import numpy as np
from scipy import stats
import collections


class DirichletMultinomial:
    __slots__ = ["alpha", "k"]

    def __init__(self, alpha=None):
        if type(alpha) == int:
            self.k = alpha
            self.alpha = np.ones(alpha)
        elif len(alpha) > 1:
            self.k = len(alpha)
            self.alpha = np.array(alpha)
        else:
            raise SyntaxError("Argument should be a vector or an int")

    def update(self, counts):
        if isinstance(counts, list):
            counts = collections.Counter(counts)
        if not isinstance(counts, dict):
            raise SyntaxError("Argument should be a dict or a list")
        counts_vec = [counts.get(i, 0) for i in range(self.k)]
        return DirichletMultinomial(np.add(self.alpha, counts_vec))

    def pdf(self, x):
        diri = stats.dirichlet(self.alpha)
        return diri.pdf(x)

    def mean(self, n=1):
        return self.alpha * n / (self.alpha.sum())

    def cdf(self, weights, x):
        Omega = lambda row: np.dot(weights, row)
        # Sample from Dirichlet posterior
        samples = np.random.dirichlet(self.alpha, 100000)
        # apply sum to sample draws
        W_samples = np.apply_along_axis(Omega, 1, samples)
        # Compute P(W > x)
        return (W_samples > x).mean()

    def posterior(self, weights, l, u):
        if l > u:
            return 0.0
        return self.cdf(weights, l) - self.cdf(weights, u)

    def sample(self, n):
        raise NotImplementedError()
