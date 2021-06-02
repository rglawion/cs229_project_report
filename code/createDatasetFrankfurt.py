# %%
import pandas as pd
import numpy as np
from scipy.stats import lognorm
import datetime as dt
import os
import seaborn as sns
from numba import jit
import matplotlib.pyplot as plt

import tqdm
import yfinance as yf


os.chdir("C:/Users/Rene/Dropbox/rene/cs229/project")

# %% Read Data
data = pd.read_parquet("data/databaseFrankfurt.parquet")
# %%
VWData = data[data["Ticker"] == "VOW.F"]
data = data[data["Ticker"] != "VOW.F"]
stockDataGrouped = data.groupby(['Ticker'])

# %%
'''
1. Create Dataset for VW with the Date, Close, Volume, and Ticker
2. Add Returns to that Dataset
3. Maybe Moving Average for Volume ?
4. Add Lags for Returns 
''' 
# %% Agenda for VW
newDataset = VWData[["Close", "Volume", "Ticker"]]

newDataset["Return_Delta1_VW"] = newDataset["Close"].pct_change(periods=1).fillna(0)
#newDataset["Return_Future_Delta1_VW"] = newDataset["Close"].pct_change(periods=-1).fillna(0)

for i in range(2,29):
    newDataset["Return_Delta{}_VW".format(i)] = 0.
    #newDataset["Return_Future_Delta{}_VW".format(i)] = 0. # newDataset["Close"].pct_change(periods=-i)

for i in range(1,28):
    newDataset["Return_Delta{}_VW".format(i+1)][i:] = newDataset["Return_Delta1_VW"][0:-i]
    #newDataset["Return_Future_Delta{}_VW".format(i+1)][0:-i] = newDataset["Return_Future_Delta1_VW"][i:]
    
for i in range(1,29):
    newDataset["Return_Future_Delta{}_VW".format(i)] = newDataset["Close"].pct_change(periods=-i).fillna(0)


# %%
for key, stockData in stockDataGrouped:
    ticker = stockData["Ticker"].iat[0]

    tmpDataFrame = stockData[["Close", "Volume", "Ticker"]]
    tmpDataFrame.columns = ["Close_{}".format(ticker), "Volume_{}".format(ticker), ticker]

    # Add return
    tmpDataFrame["Return_Delta1_{}".format(ticker)] = tmpDataFrame["Close_{}".format(ticker)].pct_change(periods=1)
    for i in range(2,29):
        tmpDataFrame["Return_Delta{}_{}".format(i, ticker)] = 0.
    
    for i in range(1,28):
        tmpDataFrame["Return_Delta{}_{}".format(i+1, ticker)][i:] = tmpDataFrame["Return_Delta1_{}".format(ticker)][0:-i]

    newDataset = pd.merge(newDataset, tmpDataFrame,  how='left', left_index=True, right_index=True)

print(newDataset)

newDataset.to_parquet("data/databaseMergedFrankfurt.parquet", engine = "fastparquet", compression = "gzip")
# %%
