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

from random import randint
from time import sleep


os.chdir("C:/Users/Rene/Dropbox/rene/cs229/project")

# %% Read Historical Data of Volkswagen
VW = yf.Ticker("VOW.F")
databaseFrankfurt = VW.history(start="2000-01-01", end="2021-04-20", interval="1d")

databaseFrankfurt["Ticker"] = "VOW.F"   # ADD Ticker for database
databaseFrankfurt["Currency"] = VW.info["currency"]
databaseFrankfurt["Exchange"] = VW.info["exchange"]
databaseFrankfurt["LongName"] = VW.info["longName"]
databaseFrankfurt["exchangeTimezoneName"] = VW.info["exchangeTimezoneName"]
databaseFrankfurt["gmtOffSetMilliseconds"] = VW.info["gmtOffSetMilliseconds"]
databaseFrankfurt["market"] = VW.info["market"]
# %% Read Supply Chain 

tickerList = pd.read_excel("data/VWSupplyChain.xlsx", sheet_name="TickerFrankfurt")
tickers = tickerList["Yahoo Finance Ticker"].dropna()

# %%
for i, ticker in enumerate(tickers):
    thisTicker = yf.Ticker(ticker)
    thisTickerHistorical = thisTicker.history(start="2000-01-01", end="2021-04-20", interval="1d")

    thisTickerHistorical["Ticker"] = ticker   # ADD Ticker for database
    thisTickerHistorical["Currency"] = thisTicker.info["currency"]
    thisTickerHistorical["Exchange"] = thisTicker.info["exchange"]
    thisTickerHistorical["LongName"] = thisTicker.info["longName"]
    thisTickerHistorical["exchangeTimezoneName"] = thisTicker.info["exchangeTimezoneName"]
    thisTickerHistorical["gmtOffSetMilliseconds"] = thisTicker.info["gmtOffSetMilliseconds"]
    thisTickerHistorical["market"] = thisTicker.info["market"]

    databaseFrankfurt = pd.concat([databaseFrankfurt, thisTickerHistorical])

    recommendations = thisTicker.recommendations
    quarterly_financials = thisTicker.quarterly_financials
    quarterly_balance_sheet = thisTicker.quarterly_balance_sheet

    # sleep random time to avoid blocking
    sleep(randint(10,100))

# save database
databaseFrankfurt.to_parquet("data/databaseFrankfurt.parquet", engine = "fastparquet", compression = "gzip")
# %%
