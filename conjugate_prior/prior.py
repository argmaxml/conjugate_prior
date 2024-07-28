from typing import List, Tuple
import numpy as np
import scipy.stats as stats
import pandas as pd
from .gamma import GammaExponential
from .normal import NormalNormalKnownVar
from .beta import BetaBinomial
from .invgamma import InvGammaWeibullKnownShape
try:
    from matplotlib import pyplot as plt
except ModuleNotFoundError:
    import sys

    sys.stderr.write("matplotlib was not found, plotting would raise an exception.\n")
    plt = None

def aic_bic(data, dist_name, params):
    """Calculate AIC and BIC for a given distribution"""
    dist = getattr(stats, dist_name)
    log_likelihood = np.sum(dist.logpdf(data, *params))
    k = len(params)
    n = len(data)
    aic = 2*k - 2*log_likelihood
    bic = np.log(n)*k - 2*log_likelihood
    return aic, bic

class ConjugatePrior:
    def __init__(self, criterion='aic') -> None:
        self.criterion = criterion
        self.best_fit = None
        self.params = {}

    def fit(self, X):
        norm_params = stats.norm.fit(X)
        exp_params = stats.expon.fit(X)
        weibull_params = stats.weibull_min.fit(X)

        # Calculate AIC and BIC for each distribution
        aic_norm, bic_norm = aic_bic(X, 'norm', norm_params)
        aic_exp, bic_exp = aic_bic(X, 'expon', exp_params)
        aic_weibull, bic_weibull = aic_bic(X, 'weibull_min', weibull_params)

        # Collect results in a DataFrame for comparison
        results = pd.DataFrame({
            'Distribution': ['norm', 'expon', 'weibull_min'],
            'AIC': [aic_norm, aic_exp, aic_weibull],
            'BIC': [bic_norm, bic_exp, bic_weibull]
        })

        if self.criterion.lower() == 'aic':
            self.best_fit = results.loc[results['AIC'].idxmin()]["Distribution"]
        elif self.criterion.lower() == 'bic':
            self.best_fit = results.loc[results['BIC'].idxmin()]["Distribution"]
        else:
            raise ValueError("Criterion must be either 'aic' or 'bic'")
        if self.best_fit == 'norm':
            self.params = norm_params
        elif self.best_fit == 'expon':
            self.params = exp_params
        elif self.best_fit == 'weibull_min':
            self.params = weibull_params
    
    def predict(self, X):
        if self.best_fit is None:
            raise ValueError("You must call the fit method first")
        dist = getattr(stats, self.best_fit)
        return dist.pdf(X, *self.params)
    
    def as_prior(self):
        # TODO: Check
        if self.best_fit == 'norm':
            return NormalNormalKnownVar(*self.params)
        elif self.best_fit == 'expon':
            return GammaExponential(*self.params)
        elif self.best_fit == 'weibull_min':
            return InvGammaWeibullKnownShape(*self.params)
        
class BetaBinomialRanker:
    def __init__(self, n=0, prior=None,ucb_percentile = 0.95, discount_coefficient=1, names=None) -> None:
        self.cmpgns = [BetaBinomial(prior) for _ in range(n)]
        self.n = n
        self.prior = prior
        self.ucb_percentile = ucb_percentile
        self.discount_coefficient = discount_coefficient
        if names is None:
            self.names = [str(i) for i in range(n)]
        else:
            self.names = names
    def __getitem__(self, name):
        i = self.names.index(name)
        return self.cmpgns[i]
    def __setitem__(self, name, value):
        try:
            i = self.names.index(name)
        except ValueError:
            self.names.append(name)
            self.cmpgns.append(BetaBinomial(self.prior))
            self.n += 1
            i = self.n - 1
        p,n = value
        self.cmpgns[i].positives = p
        self.cmpgns[i].negatives = n
    def __delitem__(self, name):
        i = self.names.index(name)
        del self.cmpgns[i]
        del self.names[i]
        self.n -= 1
    def __str__(self):
        return str(self.cmpgns)
    def reset(self):
        self.cmpgns = [BetaBinomial(self.prior) for _ in range(self.n)]
    def update(self, name, p, n):
        i = self.names.index(name)
        self.cmpgns[i] = self.cmpgns[i].update(p,n)
    def update_all(self, data: List[Tuple[int, int]]):
        assert len(data) == self.n, "Data must have the same number of campaigns as the model"
        for i, d in enumerate(data):
            p,n = d
            self.cmpgns[i] = self.cmpgns[i].update(p,n)
    def rank_by_mle(self):
        lst = sorted([(c.mean(), i) for i,c in enumerate(self.cmpgns)])
        return [self.names[i] for _,i in lst]
    def rank_by_ucb(self):
        lst = sorted([(c.percentile(self.ucb_percentile), i) for i,c in enumerate(self.cmpgns)])
        return [self.names[i] for _,i in lst]
    def discount(self):
        for i in range(self.n):
            self.cmpgns[i].positives *= self.discount_coefficient
            self.cmpgns[i].negatives *= self.discount_coefficient
        return self

class GammaExponentialRanker:
    def __init__(self, n=0, prior=None,ucb_percentile = 0.95, discount_coefficient=1, names=None) -> None:
        self.cmpgns = [GammaExponential(prior) for _ in range(n)]
        self.n = n
        self.prior = prior
        self.ucb_percentile = ucb_percentile
        self.discount_coefficient = discount_coefficient
        if names is None:
            self.names = [str(i) for i in range(n)]
        else:
            self.names = names
    def __getitem__(self, name):
        i = self.names.index(name)
        return self.cmpgns[i]
    def __setitem__(self, name, value):
        try:
            i = self.names.index(name)
        except ValueError:
            self.names.append(name)
            self.cmpgns.append(GammaExponential(self.prior))
            self.n += 1
            i = self.n - 1
        a,b = value
        self.cmpgns[i].alpha = a
        self.cmpgns[i].beta = b
    def __delitem__(self, name):
        i = self.names.index(name)
        del self.cmpgns[i]
        del self.names[i]
        self.n -= 1
    def __str__(self):
        return str(self.cmpgns)
    def reset(self):
        self.cmpgns = [GammaExponential(self.prior) for _ in range(self.n)]
    def update(self, name, data):
        i = self.names.index(name)
        self.cmpgns[i] = self.cmpgns[i].update(data)
    def update_all(self, data: List[List[int]]):
        assert len(data) == self.n, "Data must have the same number of campaigns as the model"
        for i, d in enumerate(data):
            self.cmpgns[i] = self.cmpgns[i].update(d)
    def rank_by_mle(self):
        lst = sorted([(c.mean(), i) for i,c in enumerate(self.cmpgns)])
        return [self.names[i] for _,i in lst]
    def rank_by_ucb(self):
        lst = sorted([(c.percentile(self.ucb_percentile), i) for i,c in enumerate(self.cmpgns)])