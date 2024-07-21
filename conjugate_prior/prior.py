import numpy as np
import scipy.stats as stats
import pandas as pd
from .gamma import GammaExponential
from .normal import NormalNormalKnownVar
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