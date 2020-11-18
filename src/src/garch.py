# -*- coding: utf-8 -*-

import statsmodels.tsa.api as smt
import statsmodels.api as sm
from src.xyplot_core import *
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt


def shift(xs, n):
    e = np.empty_like(xs)
    if n >= 0:
        e[:n] = np.nan
        e[n:] = xs[:-n]
    else:
        e[n:] = np.nan
        e[:n] = xs[-n:]
    return e


def garch(underlying):
    """Calculates GARCH based on the underlying asset pandas dataframe"""
    r = underlying['returns']
    garch11 = arch_model(r, p=1, q=1, rescale=True)
    res = garch11.fit(update_freq=10)
    print(res.summary())  
   
    #sigma =  0.01 *np.sqrt(res.params['omega'] + res.params['alpha[1]'] * res.resid**2 + res.conditional_volatility**2 * res.params['beta[1]']) * np.sqrt(252)
    sigma =  0.01 *np.sqrt(res.params['omega'] + res.params['alpha[1]'] * res.resid**2 + shift(res.resid, 1)**2 * res.params['beta[1]']) * np.sqrt(252)
    
    underlying['vol_garch'] = sigma
    return(underlying)


import matplotlib.dates as mdates
    
def plot_garch(underlying):
       
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    years_fmt = mdates.DateFormatter('%Y')
    
    data = underlying.iloc[1000:1300, :]
    
    fig, ax = plt.subplots()
    ax.plot('date_cboe', 'vol_garch', data=data, c='k', label='$\sigma_{GARCH(1,1)}$', linewidth=1.0)
    ax.plot('date_cboe', 'volatility5', data=data, c='r', label='$\sigma_{MA5}$', linewidth=1.0)

    # format the ticks
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(months)
    
    # format the coords message box
    ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax.format_ydata = lambda x: '$%1.2f' % x  # format the price.
    ax.grid(True)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Volatility')
    
    ax.set_title('Volatility comparision')
    ax.legend()
    
    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    fig.autofmt_xdate()
    fig.savefig("..\images\\"+'volatility_comparision.png', format="png",
                        dpi=200, bbox_inches='tight')
    
    plt.show()

    # """Plots volatility (GARCH and MA5 volatility)"""
    # plot1 = MyPlot()
    # plot1.append_data(underlying['day'], underlying['vol_garch'],
    #                   'r', '$\sigma_{GARCH(1,1)}$', linewidth=1.0)
    # plot1.append_data(
    #     underlying['day'], underlying['volatility5'], 'k', '$\sigma_{MA5}$', linewidth=2.0)
    # plot1.construct_plot("Volatility", "Date", "Volatility ($\sigma$)", save="volatility_garch.png",
    #                      xticks_bool=True, xymin=[0, 0], xymax=[4699, 1], figsize=(10, 5))
    

