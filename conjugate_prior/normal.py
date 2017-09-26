import numpy as np
from scipy import stats
from scipy import special as fn
from matplotlib import pyplot as plt
import collections, itertools
from operator import itemgetter as at

class NormalNormalKnownVar:
    __slots__ = ["mean", "var", "data_var"]
    def __init__(self, *args):
        self.mean = 0
        self.data_var = 0
        if not any(args):
            self.var = 1
        elif len(args)==1:
            self.var = args[0]
        elif len(args)==2:
            self.mean = args[0]
            self.var = args[1]
        elif len(args)==3:
            self.mean = args[0]
            self.var = args[1]
            self.data_var = args[2]
        else:
            raise SyntaxError("Illegal number of arguments")
    def update(self, data):
        var = np.var(data)
        mean = np.mean(data)
        n = len(data)
        denom = (1/(self.var * n) + 1/var)
        return NormalNormalKnownVar((self.mean/(self.var * n) + mean/var) / denom, 1.0 / (denom * n), self.var + var)
    def cdf(self, x):
        return stats.norm.cdf(x, self.mean, self.var)
    def posterior(self, l, u):
        if l>u:
            return 0.0
        return self.cdf(u)-self.cdf(l)
    def plot(self, l=0.0, u=1.0):
        x = np.linspace(u, l, 1001)
        y = stats.norm.pdf(x,self.mean, self.var)
        y=y/y.sum()
        plt.plot(x, y)
        plt.xlim((l,u))
    def predict(self, x):
        return stats.norm.cdf(x, self.mean, self.var + self.data_var)
