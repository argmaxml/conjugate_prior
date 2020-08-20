import numpy as np
from scipy import stats

try:
    from matplotlib import pyplot as plt
except ModuleNotFoundError:
    import sys

    sys.stderr.write("matplotlib was not found, plotting would raise an exception.\n")
    plt = None


class GammaExponential:
    __slots__ = ["alpha", "beta"]

    def __init__(self, alpha, beta=None):
        if beta is None:
            print("Assuming first parameter is the Expectancy")
            lamda = 1.0 / alpha
            beta = 0.5
            alpha = lamda * beta
        self.alpha = alpha
        self.beta = beta

    def update(self, *args):
        if len(args) == 1:
            return GammaExponential(self.alpha + len(args[0]), self.beta + sum(args[0]))
        elif len(args) == 2:
            return GammaExponential(self.alpha + args[0], self.beta + args[1])
        else:
            raise SyntaxError("Illegal number of arguments")

    def pdf(self, x):
        return stats.gamma.pdf(1.0 / x, self.alpha, scale=1.0 / self.beta)

    def cdf(self, x):
        return 1 - stats.gamma.cdf(1.0 / x, self.alpha, scale=1.0 / self.beta)

    def posterior(self, l, u):
        if l > u:
            return 0.0
        return self.cdf(u) - self.cdf(l)

    def mean(self):
        return self.alpha / self.beta

    def plot(self, l=0, u=10):
        x = np.linspace(u, l, 1001)
        y = stats.gamma.pdf(x, self.alpha, scale=1.0 / self.beta)
        plt.plot(x, y)
        plt.xlim((l, u))

    def plot_inverse_lambda(self, l=0.0001, u=0.999):
        x = np.linspace(1.0 / u, 1.0 / l, 1001)
        y = stats.gamma.pdf(x, self.alpha, scale=1.0 / self.beta)
        x = 1 / x
        y = list(reversed(y))
        plt.plot(x, y)
        plt.xlim((l, u))

    def predict(self, x):
        return stats.lomax.cdf(1.0 / x, self.alpha, scale=1.0 / self.beta)

    def sample(self):
        lamda = np.random.gamma(self.alpha, 1/self.beta)
        return np.random.exponential(1/lamda)


class GammaPoisson(GammaExponential):
    def update(self, *args):
        if len(args) == 1:
            return GammaPoisson(self.alpha + sum(args[0]), self.beta + len(args[0]))
        elif len(args) == 2:
            return GammaPoisson(self.alpha + args[0], self.beta + args[1])
        else:
            raise SyntaxError("Illegal number of arguments")

    def predict(self, x):
        return stats.nbinom.pmf(x, self.alpha, scale=1.0 / (1 + self.beta))

    def sample(self):
        lamda = np.random.gamma(self.alpha, 1/self.beta)
        return np.random.poisson(lamda)
