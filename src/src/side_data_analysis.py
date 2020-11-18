import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.xyplot_core import *


def plot_particular_option(pdata, strike, cpflag):
    """plots particular call option (X=1500, maturity less than 100 days for example)"""
    strike_bool = pdata['strike'] == strike
    iscall = pdata['cpflag'] == cpflag

    shortmaturity1 = pdata['maturity'] < 150
    shortmaturity2 = pdata['maturity'] < 120
    shortmaturity3 = pdata['maturity'] < 90
    shortmaturity4 = pdata['maturity'] < 60
    shortmaturity5 = pdata['maturity'] < 30

    ss1 = pdata[strike_bool & iscall & shortmaturity1]
    ss2 = pdata[strike_bool & iscall & shortmaturity2]
    ss3 = pdata[strike_bool & iscall & shortmaturity3]
    ss4 = pdata[strike_bool & iscall & shortmaturity4]
    ss5 = pdata[strike_bool & iscall & shortmaturity5]

    if cpflag == "C":
        name = "Call option price distribution (X="+str(strike)+")"
    else:
        name = "Put option price distribution (X="+str(strike)+")"

    plot1 = MyPlot()
    plot1.append_data(ss2['close'], ss2['mid'], 'k', '90<T<120', linewidth=1.5)
    plot1.append_data(ss3['close'], ss3['mid'], 'r', '60<T<90', linewidth=1.5)
    plot1.append_data(ss4['close'], ss4['mid'], 'b', '30<T<60', linewidth=1.5)
    plot1.append_data(ss5['close'], ss5['mid'], 'g', 'T<30', linewidth=1.5)
    plot1.construct_plot(name, "$S$", "$"+cpflag+"$", save=str(cpflag) +
                         str(strike)+".png", xymin=[1050, 0], xymax=[1550, 200], scatter=True)


def plot_particular_option(pdata, strike, cpflag):
    """plot call options with fixed strike price
    (X=1500, maturity less than 100 days for example)"""

    strike_bool = pdata['strike'] == strike
    iscall = pdata['cpflag'] == cpflag

    shortmaturity1 = pdata['maturity'] < 150
    shortmaturity2 = pdata['maturity'] < 120
    shortmaturity3 = pdata['maturity'] < 90
    shortmaturity4 = pdata['maturity'] < 60
    shortmaturity5 = pdata['maturity'] < 30

    ss1 = pdata[strike_bool & iscall & shortmaturity1]
    ss2 = pdata[strike_bool & iscall & shortmaturity2]
    ss3 = pdata[strike_bool & iscall & shortmaturity3]
    ss4 = pdata[strike_bool & iscall & shortmaturity4]
    ss5 = pdata[strike_bool & iscall & shortmaturity5]

    if cpflag == "C":
        name = "Call option price distribution (K="+str(strike)+")"
    else:
        name = "Put option price distribution (K="+str(strike)+")"

    plot1 = MyPlot()
    plot1.append_data(ss2['close'], ss2['mid'], 'k', '90<T<120', linewidth=1.5)
    plot1.append_data(ss3['close'], ss3['mid'], 'r', '60<T<90', linewidth=1.5)
    plot1.append_data(ss4['close'], ss4['mid'], 'b', '30<T<60', linewidth=1.5)
    plot1.append_data(ss5['close'], ss5['mid'], 'g', 'T<30', linewidth=1.5)
    plot1.construct_plot(name, "$S$", "$"+cpflag+"$", save=str(cpflag) +
                         str(strike)+".png", xymin=[1050, 0], xymax=[1550, 200], scatter=True)


def plot_particular_moneyness(pdata, cpflag):
    """plot call options with fixed moneyness 
    (maturity less than 100 days for example)"""
    moneyness1 = pdata['moneyness'] < 0.97
    moneyness2 = pdata['moneyness'] > 0.97
    moneyness2b = pdata['moneyness'] <= 1.05
    moneyness3 = pdata['moneyness'] > 1.05
    iscall = pdata['cpflag'] == cpflag

    shortmaturity1 = pdata['maturity'] >= 180
    shortmaturity2 = pdata['maturity'] < 180
    shortmaturity3 = pdata['maturity'] < 60

    ss1 = pdata[moneyness1 & iscall & shortmaturity1]
    ss2 = pdata[moneyness1 & iscall & shortmaturity2]
    ss3 = pdata[moneyness1 & iscall & shortmaturity3]

    ss4 = pdata[moneyness2 & moneyness2b & iscall & shortmaturity1]
    ss5 = pdata[moneyness2 & moneyness2b & iscall & shortmaturity2]
    ss6 = pdata[moneyness2 & moneyness2b & iscall & shortmaturity3]

    ss7 = pdata[moneyness3 & iscall & shortmaturity1]
    ss8 = pdata[moneyness3 & iscall & shortmaturity2]
    ss9 = pdata[moneyness3 & iscall & shortmaturity3]

    if cpflag == "C":
        name = "Call option moneyness distribution"
    else:
        name = "Put option moneyness distribution"

    plot1 = MyPlot()
    plot1.append_data(ss1['moneyness'], ss1['mid_strike'],
                      '#550099', '$T\geq180$', linewidth=1.0)
    plot1.append_data(ss2['moneyness'], ss2['mid_strike'],
                      '#000099', '$180>T\geq60$', linewidth=1.0)
    plot1.append_data(ss3['moneyness'], ss3['mid_strike'],
                      '#009999', '$T<60$', linewidth=1.0)

    plot1.append_data(ss4['moneyness'], ss4['mid_strike'],
                      '#7700cc', '', linewidth=1.0)
    plot1.append_data(ss5['moneyness'], ss5['mid_strike'],
                      '#0000cc', '', linewidth=1.0)
    plot1.append_data(ss6['moneyness'], ss6['mid_strike'],
                      '#00cccc', '', linewidth=1.0)

    plot1.append_data(ss7['moneyness'], ss7['mid_strike'],
                      '#9900ee', '', linewidth=1.0)
    plot1.append_data(ss8['moneyness'], ss8['mid_strike'],
                      '#0000ee', '', linewidth=1.0)
    plot1.append_data(ss9['moneyness'], ss9['mid_strike'],
                      '#00eeee', '', linewidth=1.0)

    vlines = [0.97, 1.05]
    plot1.construct_plot(name, "Moneyness $S/X$", "$"+cpflag+"/X$", save="Moneyness_"+str(
        cpflag)+".png", xymin=[0.4, 0.0], xymax=[1.6, 0.7], scatter=True, vlines=vlines)


def plot_close(underlying):
    """Plot closing prices of underlying asset"""
    plot2 = MyPlot()
    plot2.append_data(underlying['day'], underlying['close'],
                      'k', 'S&P500 daily closing price', linewidth=1.0)
    plot2.construct_plot("S&P500 daily closing price", "Date", "S", save="close.png",
                         xticks_bool=True, xymin=[0, 500], xymax=[4698, 2000], figsize=(10, 5))


def plot_returns(underlying):
    """Plot returns of underlying asset"""
    plot2 = MyPlot()
    plot2.append_data(
        underlying['day'], underlying['returns'], 'k', 'Returns', linewidth=1.0)
    plot2.construct_plot("Returns", "Date", "Returns", save="returns.png", xticks_bool=True, xymin=[
                         0, -0.15], xymax=[4698, 0.15], figsize=(10, 5))

def plot_interest(treasury):
    """Plot returns of underlying asset"""
    plot2 = MyPlot()
    plot2.append_data(
        treasury['day'], treasury['interest'], 'k', 'US 3-month treasury bond yield', linewidth=1.0)
    plot2.plot_interest("Yield", "Date", "Yield", save="yield.png", xticks_bool=True, xymin=[
                         0, -0.00], xymax=[4957, 0.065], figsize=(10, 5))

def plot_volatilities(underlying):
    """Plots different time window volatilities for underlying asset"""
    plot2 = MyPlot()
    
    plot2.append_data(
        underlying['day'],
        underlying['volatility5'],
        'k',
        '$\sigma_{MA5}$',
        linewidth=1.0)
    
    # plot2.append_data(
    #     underlying['day'],
    #     underlying['volatility30'],
    #     'r',
    #     '$\sigma_{MA30}$',
    #     linewidth=1.5)
    
    plot2.append_data(
        underlying['day'],
        underlying['volatility60'],
        'r',
        '$\sigma_{MA60}$',
        linewidth=2.0)
    
    # plot2.append_data(
    #     underlying['day'],
    #     underlying['volatility90'],
    #     'r',
    #     '$\sigma_{MA90}$',
    #     linewidth=2.5)
    
    plot2.append_data(
        underlying['day'],
        underlying['volatility120'],
        'b',
        '$\sigma_{MA120}$',
        linewidth=3.0)
    
    plot2.construct_plot("Volatility", "Date", "Volatility ($\sigma$)", save="volatility.png",
                         xticks_bool=True, xymin=[0, -0.1], xymax=[4698, 1.4], figsize=(10, 5))


def plot_black_scholes_prediction(pdata):
    """Plots Black Scholes formula prediction"""
    pdata_plt = pdata[:100000]
    plt.scatter(pdata_plt['mid'], pdata_plt['BS5'])
    plt.scatter(pdata_plt['mid'], pdata_plt['BS30'])
    plt.scatter(pdata_plt['mid'], pdata_plt['BS60'])
    plt.scatter(pdata_plt['mid'], pdata_plt['BS90'])
    plt.scatter(pdata_plt['mid'], pdata_plt['BS120'])
    plt.scatter(pdata_plt['mid'], pdata_plt['BSgarch'])
    plt.show()


def summary_table(calls_mod):
    """Summarizes table of prices for each module 
    (order:  0.97, t1,t2,t3, 1.00 t1,t2,t3, 1.05 t1,t2,t3)"""
    calls_mod_mids = [calls_mod[i]['mid'].mean() for i in range(9)]
    calls_mod_counts = [len(calls_mod[i]) for i in range(9)]
    print(calls_mod_mids, calls_mod_counts)
    calls_mod_mids_means = [calls_mod[0+i]['mid'].append(calls_mod[1+i]['mid'], ignore_index=True).append(
        calls_mod[2+i]['mid'], ignore_index=True).mean() for i in range(0, 9, 3)]
    print(calls_mod_mids_means)


def volatility_table(calls_mod):
    """Summarizes table of volatility for each module 
    (order:  0.97, t1,t2,t3, 1.00 t1,t2,t3, 1.05 t1,t2,t3)"""
        
    calls_mod_vol5 = [calls_mod[i]['volatility5'].mean()
                      for i in range(0, 9)]
    print("\\% & ".join([str(round(item*100, 2)) for item in calls_mod_vol5]))
    
    calls_mod_vol20 = [calls_mod[i]['volatility30'].mean()
                       for i in range(0, 9)]
    print("\\% & ".join([str(round(item*100, 2)) for item in calls_mod_vol30]))
    
    calls_mod_vol60 = [calls_mod[i]['volatility60'].mean()
                       for i in range(0, 9)]
    print("\\% & ".join([str(round(item*100, 2)) for item in calls_mod_vol60]))
    
    calls_mod_vol100 = [calls_mod[i]['volatility90'].mean()
                        for i in range(0, 9)]
    print("\\% & ".join([str(round(item*100, 2)) for item in calls_mod_vol90]))
                         
    calls_mod_vol100 = [calls_mod[i]['volatility120'].mean()
                        for i in range(0, 9)]
    print("\\% & ".join([str(round(item*100, 2)) for item in calls_mod_vol120]))
    
    calls_mod_volgarch = [calls_mod[i]['vol_garch'].mean()
                          for i in range(0, 9)]
    print("\\% & ".join([str(round(item*100, 2)) for item in calls_mod_volgarch]))


    calls_mod_vol5_means = [calls_mod[0+i]['volatility5'].append(calls_mod[1+i]['volatility5'], ignore_index=True).append(
        calls_mod[2+i]['volatility5'], ignore_index=True).mean() for i in range(0, 9, 3)]
    
    calls_mod_vol20_means = [calls_mod[0+i]['volatility30'].append(calls_mod[1+i]['volatility30'], ignore_index=True).append(
        calls_mod[2+i]['volatility30'], ignore_index=True).mean() for i in range(0, 9, 3)]
    
    calls_mod_vol60_means = [calls_mod[0+i]['volatility60'].append(calls_mod[1+i]['volatility60'], ignore_index=True).append(
        calls_mod[2+i]['volatility60'], ignore_index=True).mean() for i in range(0, 9, 3)]
    
    calls_mod_vol100_means = [calls_mod[0+i]['volatility90'].append(calls_mod[1+i]['volatility90'], ignore_index=True).append(
        calls_mod[2+i]['volatility90'], ignore_index=True).mean() for i in range(0, 9, 3)]
    
    calls_mod_vol100_means = [calls_mod[0+i]['volatility120'].append(calls_mod[1+i]['volatility120'], ignore_index=True).append(
        calls_mod[2+i]['volatility120'], ignore_index=True).mean() for i in range(0, 9, 3)]
    
    calls_mod_volgarch_means = [calls_mod[0+i]['vol_garch'].append(calls_mod[1+i]['vol_garch'], ignore_index=True).append(
        calls_mod[2+i]['vol_garch'], ignore_index=True).mean() for i in range(0, 9, 3)]
    
    # calls_mod_vol5_means,calls_mod_vol20_means,calls_mod_vol60_means,calls_mod_vol100_means)
    print(calls_mod_implied_vol_means, calls_mod_volgarch_means)
