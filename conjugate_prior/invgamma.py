import numpy as np
from scipy import stats

try:
    from matplotlib import pyplot as plt
except ModuleNotFoundError:
    import sys

    sys.stderr.write("matplotlib was not found, plotting would raise an exception.\n")
    plt = None


class InvGammaNormalKnownMean:
    __slots__ = ["alpha", "beta", "shape"]

    def __init__(self, *args):
        self.beta = 1
        self.shape = 1
        if len(args) == 1:
            self.alpha = args[0]
        elif len(args) == 2:
            self.alpha = args[0]
            self.beta = args[1]
        elif len(args) == 3:
            self.alpha = args[0]
            self.beta = args[1]
            self.shape = args[2]
        else:
            raise SyntaxError("Illegal number of arguments")

    def update(self, data):
        var = np.var(data)
        mean = np.mean(data)
        n = len(data)
        beta_update = sum([(d - mean) ** 2 for d in data]) / 2.0
        return InvGammaNormalKnownMean(self.alpha + n / 2.0, self.beta + beta_update)

    def pdf(self, x):
        return stats.invgamma.pdf(x, a=self.alpha, scale=self.beta)

    def cdf(self, x):
        return stats.invgamma.cdf(x, a=self.alpha, scale=self.beta)

    def posterior(self, l, u):
        if l > u:
            return 0.0
        return self.cdf(u) - self.cdf(l)

    def plot(self, l=0.0, u=3.0):
        x = np.linspace(u, l, 1001)
        y = stats.invgamma.pdf(x, a=self.alpha, scale=self.beta)
        y = y / y.sum()
        plt.plot(x, y)
        plt.xlim((l, u))

    def sample(self, n):
        raise NotImplementedError()


class InvGammaWeibullKnownShape(InvGammaNormalKnownMean):
    def update(self, data):
        return InvGammaWeibullKnownShape(self.alpha + len(data), self.beta + sum([d ** self.shape for d in data]))

    def sample(self, n):
        raise NotImplementedError()