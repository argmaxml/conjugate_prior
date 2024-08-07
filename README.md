# Conjugate Prior
Python implementation of the conjugate prior table for Bayesian Statistics

[![Downloads](http://pepy.tech/badge/conjugate-prior)](http://pepy.tech/count/conjugate-prior)

See wikipedia page:

https://en.wikipedia.org/wiki/Conjugate_prior#Table_of_conjugate_distributions

## Installation:
`pip install conjugate-prior`

## Supported Models:
  1. `BetaBinomial` - Useful for independent trials such as click-trough-rate (ctr), web visitor conversion.
  1. `BetaBernoulli` - Same as above.
  1. `GammaExponential` - Useful for churn-rate analysis, cost, dwell-time.
  1. `GammaPoisson` - Useful for time passed until event, as above.
  1. `NormalNormalKnownVar` - Useful for modeling a centralized distribution with constant noise.
  1. `NormalLogNormalKnownVar` - Useful for modeling a Length of a support phone call.
  1. `InvGammaNormalKnownMean` - Useful for modeling the effect of a noise.
  1. `InvGammaWeibullKnownShape` - Useful for reasoning about particle sizes over time.
  1. `DirichletMultinomial` - Extension of BetaBinomial to more than 2 types of events (Limited support).

## Basic API
  1. `model = GammaExponential(a, b)` - A Bayesian model with an `Exponential` likelihood, and a `Gamma` prior. Where `a` and `b` are the prior parameters.
  1. `model.pdf(x)` - Returns the probability-density-function of the prior function at `x`.
  1. `model.cdf(x)` - Returns the cumulative-density-function of the prior function at `x`.
  1. `model.mean()` - Returns the prior mean.
  1. `model.plot(l, u)` - Plots the prior distribution between `l` and `u`.
  1. `model.posterior(l, u)` - Returns the credible interval on `(l,u)` (equivalent to `cdf(u)-cdf(l)`).
  1. `model.update(data)` - Returns a *new* model after observing `data`.
  1. `model.predict(x)` - Predicts the likelihood of observing `x` (if a posterior predictive exists).
  1. `model.sample()` - Draw a single sample from the posterior distribution.



## Coin flip example:

    from conjugate_prior import BetaBinomial
    heads = 95
    tails = 105
    prior_model = BetaBinomial() # Uninformative prior
    updated_model = prior_model.update(heads, tails)
    credible_interval = updated_model.posterior(0.45, 0.55)
    print ("There's {p:.2f}% chance that the coin is fair".format(p=credible_interval*100))
    predictive = updated_model.predict(50, 50)
    print ("The chance of flipping 50 Heads and 50 Tails in 100 trials is {p:.2f}%".format(p=predictive*100))

## Variant selection with Multi-armed-bandit

Assume we have `10` creatives (variants) we can choose for our ad campaign, at first we start with the uninformative prior.

After getting feedback (i.e. clicks) from displaying the ads, we update our model.

Then we sample the `DirrechletMultinomial` model for the updated distribution.

    from conjugate_prior import DirichletMultinomial
    from collections import Counter
    # Assuming we have 10 creatives
    model = DirichletMultinomial(10)
    mle = lambda M:[int(r.argmax()) for r in M]
    selections = [v for k,v in sorted(Counter(mle(model.sample(100))).most_common())]
    print("Percentage before 1000 clicks: ",selections)
    # after a period of time, we got this array of clicks
    clicks = [400,200,100,50,20,20,10,0,0,200]
    model = model.update(clicks)
    selections = [v for k,v in sorted(Counter(mle(model.sample(100))).most_common())]
    print("Percentage after 1000 clicks: ",selections)

## Naive Recommendation System with UCB

  from conjugate_prior import BetaBinomialRanker
  ranker = BetaBinomialRanker(prior=0.1) # 10% click-through-rate
  ranker["cmpgn1"]+=(1,9) # 1 click, 9 skips
  ranker["cmpgn2"]+=(10,90) # 10 click, 90 skips
  ranker["cmpgn3"]+=(1,2) # 1 click, 3 skips
  # Balance exploration and exploitation w/UCB
  print(ranker.rank_by_ucb())