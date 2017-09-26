# conjugate_prior
Python implementation of the conjugate prior table for Bayesian Statistics


See wikipedia page:

https://en.wikipedia.org/wiki/Conjugate_prior#Table_of_conjugate_distributions

## Installation:
`pip install conjugate_prior`

## Supported Models:
  1. BetaBinomial - Useful for independent trials such as click-trough-rate (ctr), web visitor conversion.
  1. GammaExponential - Useful for churn-rate analysis, cost, dwell-time.

## Coin flip example:

    from conjugate_prior import BetaBinomial
    heads = 100
    tails = 200
    prior_model = BetaBinomial() #Uninformative prior
    updated_model = prior_model.update(heads, tails)
    credible_interval = updated_model.posterior(0.45, 0.55)
    print ("There's {p:.2f}% chance that the coin is fair".format(p=credible_interval*100))
