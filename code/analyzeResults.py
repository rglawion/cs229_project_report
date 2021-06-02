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
from dateutil.relativedelta import *

os.chdir("C:/Users/rene/Dropbox/rene/cs229/project")

train_results = pd.read_parquet("results/Summary_Results_Trainset.parquet")
val_results = pd.read_parquet("results/Summary_Results_Valset.parquet")
test_results = pd.read_parquet("results/Summary_Results_Testset.parquet")

# %%
train_results_agg= test_results.groupby(['model','day']).agg(np.mean)
val_results_agg= test_results.groupby(['model','day']).agg(np.mean)
test_results_agg= test_results.groupby(['model','day']).agg(np.mean)

pd.options.display.float_format = '{:,.4f}'.format
table_paper = test_results[["model", "day", "root_mean_squared_error", "mean absolute_error",  "err", "estimation_time_complete"]]
table_paper = table_paper.groupby(['model','day']).agg(np.mean)
# %%
print(train_results_agg)
print(val_results_agg)
print(test_results_agg)
# %%
test_results["err_sq"] = test_results["err"]**2
test_results["y_sq"] = test_results["y"]**2
test_results_grouped= test_results.groupby(['model']).agg(np.sum)
test_results_grouped["R2_OOS"] = (1-test_results_grouped["err_sq"]/test_results_grouped["y_sq"])


# %%

train_results = pd.read_parquet("results/Summary_Results_Trainset.parquet")
val_results = pd.read_parquet("results/Summary_Results_Valset.parquet")
test_results = pd.read_parquet("results/Summary_Results_Testset.parquet")

# %% plot VW Stock
TrainDate= datetime(2005,1,1)
EndDate= datetime(2020,2,28)
data = pd.read_parquet("data/databaseFrankfurt.parquet")
data = data[data["Ticker"]=="VOW.F"]
data = data[data.index<=EndDate]
data = data[data.index>=TrainDate].sort_index()

xmax = data.index[np.argmax(data["Close"])]
ymax = data["Close"].max()

ax = data["Close"].plot(color = ['#8C1515'], linewidth=3.0)
plt.autoscale(enable=True, axis='x', tight=True)
ax.axis(ymin=0.0)
ax.set_ylabel('Price (in Euro)', fontsize='x-large')
ax.set_xlabel('Date', fontsize='x-large')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.annotate('Short Squeeze', xy=(xmax, ymax), xytext=(xmax +relativedelta(months=24, day=31, weekday=FR(-1)), ymax-100),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.margins(0,0)
plt.savefig("VWStock.pdf", bbox_inches = 'tight', pad_inches = 0)
plt.show()
# %%
