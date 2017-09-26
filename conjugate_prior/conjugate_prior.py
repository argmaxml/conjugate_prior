import numpy as np
from scipy import stats
from scipy import special as fn
from matplotlib import pyplot as plt
import collections, itertools
from operator import itemgetter as at


class GammaExponential:
    __slots__ = ["alpha", "beta"]
    def __init__(self, alpha, beta=None):
        if beta is None:
            print ("Assuming first parameter is the Expectancy")
            lamda = 1.0/alpha
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
    def cdf(self, x):
        return 1-stats.gamma.cdf(1.0/x, self.alpha, scale=1.0/self.beta)
    def posterior(self, l, u):
        if l>u:
            return 0.0
        return self.cdf(u)-self.cdf(l)
    def mean(self):
        return self.beta / self.alpha
    def plot(self, l=0, u=10):
        x = np.linspace(u, l, 1001)
        y = stats.gamma.pdf(x,self.alpha, scale=1.0/self.beta)
        plt.plot(x, y)
        plt.xlim((l,u))
    def plot_inverse_lambda(self, l=0.0001, u=0.999):
        x = np.linspace(1.0/u, 1.0/l, 1001)
        y = stats.gamma.pdf(x,self.alpha, scale=1.0/self.beta)
        x=1/x
        y=list(reversed(y))
        plt.plot(x, y)
        plt.xlim((l,u))
    def predict(self, x):
        return stats.lomax.cdf(1.0/x, self.alpha, scale=1.0/self.beta)

class GammaPoisson(GammaExponential):
    def update(self, *args):
        if len(args) == 1:
            return GammaPoisson(self.alpha + sum(args[0])), self.beta + len(args[0])
        elif len(args) == 2:
            return GammaPoisson(self.alpha + args[0], self.beta + args[1])
        else:
            raise SyntaxError("Illegal number of arguments")
    def predict(self, x):
        return stats.nbinom.pmf(x, self.alpha, scale=1.0/(1+self.beta))

class BetaBinomial:
    __slots__ = ["T", "F"]
    def __init__(self, *args):
        if len(args)==1:
            #assuming ctr
            self.T = args[0]*100.0
            self.F = (1-args[0])*100.0
        elif len(args)==2:
            self.T = args[0]
            self.F = args[1]
        else:
            raise SyntaxError("Illegal number of arguments")
    def update(self, *args):
        if len(args)==1:
            n = p = 0
            for x in args[0]:
                if x:
                    p += 1
                else:
                    n += 1
            return BetaBinomial(self.T + p, self.F + n)
        elif len(args)==2:
            return BetaBinomial(self.T + args[0], self.F + args[1])
        else:
            raise SyntaxError("Illegal number of arguments")
    def cdf(self, x):
        return stats.beta.cdf(x, self.T, self.F)
    def posterior(self, l, u):
        if l>u:
            return 0.0
        return self.cdf(u)-self.cdf(l)
    def mean(self, n=1):
        return self.T * n / (self.T + self.F)
    def plot(self, l=0.0, u=1.0):
        x = np.linspace(u, l, 1001)
        y = stats.beta.pdf(x,self.T, self.F)
        y=y/y.sum()
        plt.plot(x, y)
        plt.xlim((l,u))
    def predict(self, k, n, log=False):
        a = self.T
        b = self.F
        log_pmf = (fn.gammaln(n+1) + fn.gammaln(k+a) + fn.gammaln(n-k+b) + fn.gammaln(a+b)) - \
        (fn.gammaln(k+1) + fn.gammaln(n-k+1) + fn.gammaln(a) + fn.gammaln(b) + fn.gammaln(n+a+b))
        if log:
            return log_pmf
        return np.exp(log_pmf)

class BetaBernoulli(BetaBinomial):
    def update(self, *args):
        if len(args)==1:
            n = p = 0
            for x in args[0]:
                if x:
                    p += 1
                else:
                    n += 1
            return BetaBernoulli(self.T + p, self.F + n)
        elif len(args)==2:
            return BetaBernoulli(self.T + args[0], self.F + args[1])
        else:
            raise SyntaxError("Illegal number of arguments")

class DirichletMultinomial:
    __slots__ = ["alpha", "k", "pdf"]
    def __init__(self, alpha=None):
        if type(alpha)==int:
            self.k = alpha
            self.alpha = np.ones(alpha)
        elif len(alpha)>1:
            self.k = len(alpha)
            self.alpha = np.array(alpha)
        else:
            raise SyntaxError("Argument should be a vector or an int")
    def update(self, counts):
        if isinstance(counts, list)
            counts = collections.Counter(counts)
        if not isinstance(counts, dict):
            raise SyntaxError("Argument should be a dict or a list")
        counts_vec = [counts.get(i,0) for i in range(self.k)]
        return DirichletMultinomial(np.add(self.alpha, counts_vec))
    def pdf(self, x):
        diri = stats.dirichlet(self.alpha)
        return diri.pdf(x)
    def mean(self, n=1):
        return self.alpha * n / (self.alpha.sum())
