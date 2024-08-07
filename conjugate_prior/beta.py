import numpy as np
from scipy import stats
from scipy import special as fn

try:
    from matplotlib import pyplot as plt
except ModuleNotFoundError:
    import sys

    sys.stderr.write("matplotlib was not found, plotting would raise an exception.\n")
    plt = None


class BetaBinomial:
    __slots__ = ["positives", "negatives"]

    def __init__(self, *args):
        if not any(args) or args[0] is None:
            # uninformative prior
            self.positives = self.negatives = 1
        elif len(args) == 1:
            # assuming rate
            self.positives = args[0] * 100.0
            self.negatives = (1 - args[0]) * 100.0
        elif len(args) == 2:
            self.positives = args[0]
            self.negatives = args[1]
        else:
            raise SyntaxError("Illegal number of arguments")

    def __iadd__(self, other):
        if isinstance(other, BetaBinomial):
            self.positives += other.positives
            self.negatives += other.negatives
        elif isinstance(other, tuple):
            self.positives += other[0]
            self.negatives += other[1]
        else:
            raise TypeError("Unsupported type")
        return self
    def update(self, *args):
        if len(args) == 1:
            n = p = 0
            for x in args[0]:
                if x:
                    p += 1
                else:
                    n += 1
            return BetaBinomial(self.positives + p, self.negatives + n)
        elif len(args) == 2:
            return BetaBinomial(self.positives + args[0], self.negatives + args[1])
        else:
            raise SyntaxError("Illegal number of arguments")

    def pdf(self, x):
        return stats.beta.pdf(x, self.positives, self.negatives)

    def cdf(self, x):
        return stats.beta.cdf(x, self.positives, self.negatives)

    def posterior(self, l, u):
        if l > u:
            return 0.0
        return self.cdf(u) - self.cdf(l)

    def mean(self, n=1):
        return self.positives * n / (self.positives + self.negatives)

    def plot(self, l=0.0, u=1.0):
        x = np.linspace(u, l, 1001)
        y = stats.beta.pdf(x, self.positives, self.negatives)
        y = y / y.sum()
        plt.plot(x, y)
        plt.xlim((l, u))

    def predict(self, t, f, log=False):
        a = self.positives
        b = self.negatives
        log_pmf = (fn.gammaln(t + f + 1) + fn.gammaln(t + a) + fn.gammaln(f + b) + fn.gammaln(a + b)) - \
                  (fn.gammaln(t + 1) + fn.gammaln(f + 1) + fn.gammaln(a) + fn.gammaln(b) + fn.gammaln(t + f + a + b))
        if log:
            return log_pmf
        return np.exp(log_pmf)

    def sample(self, n=1):
        p = np.random.beta(self.positives, self.negatives,n)
        return p
    
    def percentile(self, p):
        return stats.beta.ppf(p, self.positives, self.negatives)


class BetaBernoulli(BetaBinomial):
    def update(self, *args):
        if len(args) == 1:
            n = p = 0
            for x in args[0]:
                if x:
                    p += 1
                else:
                    n += 1
            return BetaBernoulli(self.positives + p, self.negatives + n)
        elif len(args) == 2:
            return BetaBernoulli(self.positives + args[0], self.negatives + args[1])
        else:
            raise SyntaxError("Illegal number of arguments")

    def sample(self, n=1,output_parameter=False):
        p = np.random.beta(self.positives, self.negatives,n)
        if output_parameter:
            return p
        return int(np.random.random() < p)
