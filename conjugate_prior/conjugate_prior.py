import numpy as np
from scipy import stats
from matplotlib import pyplot as plt


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


if __name__ == "__main__":
    def dataframe2stats(df):
        total_clicks = df["clicks"].sum()
        total_cost = (df["usd_per_click"] * df["clicks"]).sum() * 100.0 / total_clicks
        total_impr = df["impressions"].sum()
        return total_cost, total_clicks, total_impr

    import sys
    import pandas as pd
    fname = "histo_201708260600_201708280545.tsv"#sys.argv[1]
    reference_key = "server"
    reference_val = "BF_P2"
    test_val      = "BF_P1"
    df = pd.DataFrame.from_csv(fname, index_col=None, sep="\t")
    #df=df[(df.ts==700) & (df.dt==20170827)]
    df_reference = df[df[reference_key]==reference_val]
    df_test = df[df[reference_key]==test_val]
    ref_cost, ref_clicks, ref_impr = dataframe2stats(df_reference)
    test_cost, test_clicks, test_impr = dataframe2stats(df_test)
    ref_ctr = float(ref_clicks) / ref_impr
    print ("Cost: Ref: {r}, Test: {t}".format(r=ref_cost, t=test_cost))
    print ("Clicks: Ref: {r}, Test: {t}".format(r=ref_clicks, t=test_clicks))
    print ("Impressions: Ref: {r}, Test: {t}".format(r=ref_impr, t=test_impr))
    cost_model = GammaExponential(ref_cost).update(test_clicks, test_cost*test_clicks)
    ctr_model = BetaBinomial(ref_ctr).update(test_clicks, test_impr - test_clicks)
    cost_delta = 0.01
    ctr_delta = 0.001
    cost_model_p = cost_model.posterior(ref_cost - cost_delta, ref_cost + cost_delta)
    ctr_model_p = ctr_model.posterior(ref_ctr- ctr_delta, ref_ctr + ctr_delta)
    print ("Ctr diff = {ctr}, cost diff = {cost}".format(cost=cost_model_p, ctr=ctr_model_p))