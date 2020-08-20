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
    __slots__ = ["T", "F"]

    def __init__(self, *args):
        if not any(args):
            # uninformative prior
            self.T = self.F = 1
        elif len(args) == 1:
            # assuming rate
            self.T = args[0] * 100.0
            self.F = (1 - args[0]) * 100.0
        elif len(args) == 2:
            self.T = args[0]
            self.F = args[1]
        else:
            raise SyntaxError("Illegal number of arguments")

    def update(self, *args):
        if len(args) == 1:
            n = p = 0
            for x in args[0]:
                if x:
                    p += 1
                else:
                    n += 1
            return BetaBinomial(self.T + p, self.F + n)
        elif len(args) == 2:
            return BetaBinomial(self.T + args[0], self.F + args[1])
        else:
            raise SyntaxError("Illegal number of arguments")

    def pdf(self, x):
        return stats.beta.pdf(x, self.T, self.F)

    def cdf(self, x):
        return stats.beta.cdf(x, self.T, self.F)

    def posterior(self, l, u):
        if l > u:
            return 0.0
        return self.cdf(u) - self.cdf(l)

    def mean(self, n=1):
        return self.T * n / (self.T + self.F)

    def plot(self, l=0.0, u=1.0):
        x = np.linspace(u, l, 1001)
        y = stats.beta.pdf(x, self.T, self.F)
        y = y / y.sum()
        plt.plot(x, y)
        plt.xlim((l, u))

    def predict(self, t, f, log=False):
        a = self.T
        b = self.F
        log_pmf = (fn.gammaln(t + f + 1) + fn.gammaln(t + a) + fn.gammaln(f + b) + fn.gammaln(a + b)) - \
                  (fn.gammaln(t + 1) + fn.gammaln(f + 1) + fn.gammaln(a) + fn.gammaln(b) + fn.gammaln(t + f + a + b))
        if log:
            return log_pmf
        return np.exp(log_pmf)

    def sample(self, n):
        p = np.random.beta(self.T, self.F)


class BetaBernoulli(BetaBinomial):
    def update(self, *args):
        if len(args) == 1:
            n = p = 0
            for x in args[0]:
                if x:
                    p += 1
                else:
                    n += 1
            return BetaBernoulli(self.T + p, self.F + n)
        elif len(args) == 2:
            return BetaBernoulli(self.T + args[0], self.F + args[1])
        else:
            raise SyntaxError("Illegal number of arguments")

    def sample(self, output_parameter=False):
        p = np.random.beta(self.T, self.F)
        if output_parameter:
            return p
        return int(np.random.random() < p)
