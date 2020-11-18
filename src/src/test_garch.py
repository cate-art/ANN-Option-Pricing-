import pandas as pd
from arch import arch_model
import numpy as np

underlying = pd.read_csv('../../data/index_series.csv')


"""Calculates GARCH based on the underlying asset pandas dataframe"""
r = underlying['return']
garch11 = arch_model(r, p=1, q=1)
res = garch11.fit(update_freq=10)
print(res.summary())
params = list(res.params)
print(params)
residuals = res.resid
res2 = list(residuals*residuals)
sigma2 = np.zeros(len(residuals))

for i in range(0, len(sigma2)-1):
    sigma2[i+1] = params[1] + params[2]*res2[i] + params[3]*res.conditional_volatility[]
sigma2[0] = 0.05

sigma = np.sqrt(sigma2)*np.sqrt(252)
underlying['vol_garch'] = sigma

