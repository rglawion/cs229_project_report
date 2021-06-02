# %%
import pandas as pd
import numpy as np
from scipy.stats import lognorm
import datetime as dt
import os
import seaborn as sns
from numba import jit
import matplotlib.pyplot as plt

from datetime import datetime

# %% Read Data
database_yahoo = pd.read_parquet("data/databaseMergedFrankfurt.parquet", engine = "fastparquet")
database_Swaprates = pd.read_excel("data/relDiff_monthly_IR.xlsx")
database_MacroMonthly = pd.read_excel("data/DataEurope.xlsx", sheet_name="Monthly")
database_MacroQuarterly = pd.read_excel("data/DataEurope.xlsx", sheet_name="Quarterly")

# %% Get Main datafrane
cutoff_start = datetime(2005,1,1)
cutoff_end = datetime(2019,12,31)

database_yahoo = database_yahoo[database_yahoo.index > cutoff_start]
database_yahoo = database_yahoo[database_yahoo.index < cutoff_end]

# %% prepare for merge
database_yahoo["Date"] = pd.to_datetime(database_yahoo.index, utc = True)
database_Swaprates["Timestamp"] = pd.to_datetime(database_Swaprates["Timestamp"], utc = True)
database_MacroMonthly["date"] = pd.to_datetime(database_MacroMonthly["date"], utc = True)
database_MacroQuarterly["date"] = pd.to_datetime(database_MacroQuarterly["date"], utc = True)

database_Swaprates.rename(columns={'Timestamp':'Date'}, inplace=True)
database_MacroMonthly.rename(columns={'date':'Date'}, inplace=True)
database_MacroQuarterly.rename(columns={'date':'Date'}, inplace=True)

database_yahoo.reset_index(inplace=True, drop=True)
database_Swaprates.drop(["Unnamed: 0"], inplace=True, axis=1)
database_yahoo= database_yahoo.sort_values(['Date'], ascending=[True])
database_Swaprates= database_Swaprates.sort_values(['Date'], ascending=[True])
database_MacroMonthly= database_MacroMonthly.sort_values(['Date'], ascending=[True])
database_MacroQuarterly= database_MacroQuarterly.sort_values(['Date'], ascending=[True])

# %% Merge data
database_yahoo=pd.merge_asof(database_yahoo, database_Swaprates, on = "Date", allow_exact_matches=True)
database_yahoo=pd.merge_asof(database_yahoo, database_MacroMonthly, on = "Date", allow_exact_matches=True)
database_yahoo=pd.merge_asof(database_yahoo, database_MacroQuarterly, on = "Date", allow_exact_matches=True)

# %% Save dataset
database_yahoo.to_parquet("data/databaseFrankfurtComplete.parquet", engine = "fastparquet", compression = "gzip")