import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import src.prepare_df as prepare_df
import src.black_scholes as black_scholes
from src.garch import plot_garch

pdata = prepare_df.prepare_dataframe()
pdata = pdata[pdata['cpflag'] == 'C']

# Adds columns to the option dataframe and creates dataframe for underlying asset
pdata = prepare_df.add_risk_free_rate_from_FED_to_pdata(pdata)
underlying = prepare_df.prepare_underlying_asset(pdata)
pdata = prepare_df.append_volatility_columns(pdata, underlying)
plot_garch(underlying)

# # Takes long time - adding Black Scholes results to the dataframe
pdata = black_scholes.compute_and_append_black_scholes_columns(pdata)
pdata = black_scholes.append_moneyness_columns(pdata)

# # Export raw data file
pdata.to_csv("../data/rawdata.csv", index=False, header=True)
underlying.to_csv("../data/underlying.csv", index=False, header=True)
