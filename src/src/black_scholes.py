# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 17:58:11 2020

@author: Caterina
"""

"""Black Scholes functions"""
import scipy.stats
import numpy as np


def BS_d1(S, X, r, tau, sigma):
    """Auxiliary function for BS model"""
    d1 = (np.log(S/X)+(r+0.5*sigma**2)*tau)/(sigma*np.sqrt(tau))
    return(d1)


def BS_d2(S, X, r, tau, sigma):
    """Auxiliary function for BS model"""
    d2 = BS_d1(S, X, r, tau, sigma)-sigma*np.sqrt(tau)
    return(d2)


def BS(S, X, r, tau, sigma, cpflag):
    """Standard Black-Scholes formula for call and put options"""
    d1 = BS_d1(S, X, r, tau, sigma)
    d2 = BS_d2(S, X, r, tau, sigma)
    if cpflag == "C":
        C = S*scipy.stats.norm.cdf(d1)-X*np.exp(-r*tau) * \
            scipy.stats.norm.cdf(d2)
        return(C)
    elif cpflag == "P":
        P = -S*scipy.stats.norm.cdf(-d1)+X * \
            np.exp(-r*tau)*scipy.stats.norm.cdf(-d2)
        return(P)

# ToDo: We changed the indices in df=[1] to match the ones in our dataset
def df_BS_function(df, *args):
    """Function indexing columns in pandas dataframe"""
    S = df['close']  # close
    X = df['strike']  # strike
    r = df['interest']  # /100 #discount-monthly
    tau = df['normmat']  # "normmat"
    sigma = df[args[0]]  # volat100
    cpflag = df['cpflag']  # call or put
    C = BS(S, X, r, tau, sigma, cpflag)
    return(C)

#import copy
# ToDo: We changed the indices in args=[21] to match the ones in our dataset
def compute_and_append_black_scholes_columns(pdata):
    """Appending BS results to pandas dataframe"""
    print("INFO: Computing black and Scholes for volatility = 5")
    pdata['BS5'] = pdata.apply(df_BS_function, args=[14], axis=1)
    print("INFO: Computing black and Scholes for volatility = 30")
    pdata['BS30'] = pdata.apply(df_BS_function, args=[15], axis=1)
    print("INFO: Computing black and Scholes for volatility = 60")
    pdata['BS60'] = pdata.apply(df_BS_function, args=[16], axis=1)
    print("INFO: Computing black and Scholes for volatility = 90")
    pdata['BS90'] = pdata.apply(df_BS_function, args=[17], axis=1)
    print("INFO: Computing black and Scholes for volatility = 120")
    pdata['BS120'] = pdata.apply(df_BS_function, args=[18], axis=1)
    print("INFO: Computing black and Scholes for volatility = GARCH")
    pdata['BSgarch'] = pdata.apply(df_BS_function, args=[19], axis=1)
    return(pdata)

def append_moneyness_columns(pdata):
    """Appending moneyness results to pandas dataframe"""
    pdata['BS5-strike'] = pdata['BS5']/pdata['strike']
    pdata['BS30-strike'] = pdata['BS30']/pdata['strike']
    pdata['BS60-strike'] = pdata['BS60']/pdata['strike']
    pdata['BS90-strike'] = pdata['BS90']/pdata['strike']
    pdata['BS120-strike'] = pdata['BS120']/pdata['strike']
    pdata['BSgarch-strike'] = pdata['BSgarch']/pdata['strike']
    return(pdata)