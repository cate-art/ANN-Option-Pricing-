import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import src.garch as garch

OPTION_DATASET_FILE = '../data/option.csv'
FED_TBILLS_FILE = '../data/interest.csv'
UNDERLYING_ASSET_FILE = '../data/index.csv'

def load_data(filename):
    """Loads dataset from .csv file"""
    rows = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            rows.append(row)
    return(rows)


def transform_date_to_datetime(date):
    """Auxiliary function to create dataframe"""
    date = date.split("-")
    date = datetime.datetime(int(date[0]), int(date[1]), int(date[2]))
    return(date)


def deploy_data_structure(variables, rows):
    """Auxiliary function to create dataframe"""
    data = []
    for i in range(len(rows)):
        dict1 = dict()
        for j in range(len(rows[i])-1):
            #if j== 2 because we have the first date and j== 13 because we have the second date
            if j == 1 or j == 5:
                rows[i][j] = transform_date_to_datetime(rows[i][j])
            dict1[variables[j]] = rows[i][j]
        data.append(dict1)
    return(data)


def prepare_dataframe():
    """loads data, change date to datetime format and prepares pd.DataFrame"""
    rows = load_data(OPTION_DATASET_FILE)
    data = deploy_data_structure(rows[0], rows[1:])
    pdata = pd.DataFrame(rows[1:], columns=rows[0])

    # transform data type
    pdata[['maturity']] = pdata[['maturity']].astype("int")
    pdata[['close']] = pdata[['close']].astype("float")
    pdata[['strike']] = pdata[['strike']].astype("float")
    pdata[['bid']] = pdata[['bid']].astype("float")
    pdata[['ask']] = pdata[['ask']].astype("float")
    pdata[['optionid']] = pdata[['optionid']].astype("int")
        
    # add new columns
    pdata['normmat'] = pdata['maturity']/252
    pdata['mid'] = (pdata['bid']+pdata['ask'])/2
    pdata['moneyness'] = pdata['close']/pdata['strike']
    pdata['mid_strike'] = pdata['mid']/pdata['strike']
    return(pdata)


def add_risk_free_rate_from_FED():
    """Obtains risk free rate from FED"""
    riskfreerate = load_data(FED_TBILLS_FILE)
    #i need to drop the first column of the risk free rate because is the column index
    #riskfreerate.drop([''])
    for row in riskfreerate[1:]:
        row[0] = transform_date_to_datetime(date=row[0])
        row[1] = float(row[1])
      
    riskdf = pd.DataFrame(riskfreerate[1:], columns=riskfreerate[0])
    return(riskdf)


def add_risk_free_rate_from_FED_to_pdata(pdata):
    """Appends risk free rate column"""
    riskdf = add_risk_free_rate_from_FED()

    if 'discount-monthly' not in pdata.columns:
        pdata = pdata.join(riskdf.set_index('date-rf'), on='date')
    return(pdata)


def prepare_underlying_asset(pdata):
    """Create dataframe for the underlying asset from the option data"""
    underlying = load_data(UNDERLYING_ASSET_FILE)
    
    for row in underlying[1:]:
        row[0] = transform_date_to_datetime(date=row[0])
        row[1] = float(row[1])
        row[2] = float(row[2])
    cboe_df = pd.DataFrame(underlying[1:], columns=underlying[0])
    underlying = cboe_df
    
    #create new variables
    underlying['volatility5'] = pd.Series.rolling(
        underlying.returns, window=5).std()*np.sqrt(252)
    underlying['volatility30'] = pd.Series.rolling(
        underlying.returns, window=30).std()*np.sqrt(252)
    underlying['volatility60'] = pd.Series.rolling(
        underlying.returns, window=60).std()*np.sqrt(252)
    underlying['volatility90'] = pd.Series.rolling(
        underlying.returns, window=90).std()*np.sqrt(252)
    underlying['volatility120'] = pd.Series.rolling(
        underlying.returns, window=120).std()*np.sqrt(252)

    # ToDo: Check why this was done here
    #underlying = underlying[103:-395]  # subsetting for existing option data article page 628 [https://academic.oup.com/jfec/article/15/4/602/4055906]
    underlying['day'] = range(len(underlying))
    underlying['intercept'] = np.ones(len(underlying))
    
    underlying = garch.garch(underlying)
    return(underlying)


def append_volatility_columns(pdata, underlying):
    """Appends the volatility of underlying asset to the option dataframe"""
    if 'volatility5' not in pdata:
        pdata = pdata.join(underlying[['date_cboe', 'volatility5', 'volatility30',
                                       'volatility60','volatility90','volatility120', 'vol_garch']].set_index('date_cboe'), on='date')

    return(pdata)


