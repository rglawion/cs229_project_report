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

import time

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lightgbm

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error

from sklearn import metrics

from itertools import cycle

import warnings
warnings.filterwarnings("ignore")

os.chdir("C:/Users/Rene/Dropbox/rene/cs229/project")

# %%
df_Summary_Results_Trainset = pd.DataFrame()
df_Summary_Results_Testset = pd.DataFrame()


for day in [1, 7, 14, 21, 28]: #range(1,28):

    # Read Data
    data = pd.read_parquet("data/databaseMergedFrankfurt.parquet")

    tickerList = pd.read_excel("data/VWSupplyChain.xlsx", sheet_name="TickerFrankfurt")
    tickers = tickerList["Yahoo Finance Ticker"].dropna()
    # %% First Only work with next days Return
    #y = data["Return_Future_Delta{}_VW".format(day)]
    trainColumns = ["Return_Future_Delta{}_VW".format(day)]
    for j, ticker in enumerate(tickers):
        for i in range(1,28):
            trainColumns.append("Return_Delta{}_{}".format(i, ticker))
        
    for i in range(1,28):
        trainColumns.append("Return_Delta{}_VW".format(i))

    X = data[trainColumns]

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.model_selection import train_test_split

    trainSet, testSet = train_test_split(X, train_size=.8, random_state=42)

    y_train = trainSet["Return_Future_Delta{}_VW".format(day)].values
    y_test = testSet["Return_Future_Delta{}_VW".format(day)].values

    X_train = trainSet.drop(["Return_Future_Delta{}_VW".format(day)], axis=1)
    X_test = testSet.drop(["Return_Future_Delta{}_VW".format(day)], axis=1)

    # Add the models to the list 
    models = [
    {
        'label': 'Elastic Net',
        'estimator': ElasticNet(),
        'param_grid' : {
            'alpha' : np.logspace(-6, 6, 13),
            'l1_ratio' : [0, 0.5, 1],
            },
        'linestyle' : (0, ()),
    },
    {
        'label': 'Decision Tree Regressor',
        'estimator': DecisionTreeRegressor(),
        'param_grid' : {
            'max_depth': [4, 10, 20, 100, None],
            'min_samples_split': [1, 2, 4, 10, 20],
            'min_samples_leaf' : [1, 2, 4, 8],
            'max_features' : ['auto', 'sqrt', 'log2', 'none']
        },
        'linestyle' : (0, (1, 1)),
    },
    {
        'label': 'XGBoost',
        'estimator': xgb.XGBRegressor(),
        'param_grid' : {
            'max_depth': [3, 6, 12, 20],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [50, 100, 200],
            'gamma': [0, 0.1, 0.3],
        },
        'linestyle' : '--', # (0, (3, 1, 1, 1)),
    },
    {
        'label': 'Light GBM',
        'estimator': lightgbm.LGBMRegressor(),
        'param_grid' : {
            'max_depth': [3, 6, 12, 20],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [50, 100, 200],
            'num_leaves': [11, 31, 61, 101],
        },
        'linestyle' : '-.', # (0, (3, 5, 1, 5, 1, 5)),
    }
    ]

    # Below for loop iterates through your models list
    for j, model in enumerate(models):
        
        print("Day : {} \t model : {}".format(str(day), model['label']))
        start = time.time()
        

        estimator = model['estimator'] # select the model
        parameters = model['param_grid']

        clf = GridSearchCV(estimator, parameters, scoring='neg_root_mean_squared_error', cv=5, n_jobs=60)

        clf.fit(X_train, y_train) # train the model

        # MAYBE
        best_clf = clf.best_estimator_

        y_pred=best_clf.predict(X_train) # predict the train data

        explained_variance_score = -metrics.explained_variance_score(y_train, y_pred)
        neg_mean_absolute_error_score = metrics.mean_absolute_error(y_train, y_pred)
        neg_mean_squared_error_score = metrics.mean_squared_error(y_train, y_pred, squared=True)
        neg_root_mean_squared_error_score = metrics.mean_squared_error(y_train, y_pred, squared=False)
        r2 = -metrics.r2_score(y_train, y_pred)
        neg_mean_absolute_percentage_error_score = metrics.mean_absolute_percentage_error(y_train, y_pred)

        d = {   'day' : day,
                'model': model['label'], 
                'explained variance' : [explained_variance_score],
                'mean absolute error' : [neg_mean_absolute_error_score],
                'mean squared error' : [neg_mean_squared_error_score],
                'root mean squared error' : [neg_root_mean_squared_error_score],
                'R2' : [r2],
                'mean absolute percentage error' : [neg_mean_absolute_percentage_error_score],
                'best estimator' : [clf.best_params_],
            }
        SummaryResultsThisIteration =  pd.DataFrame(data=d) 
        df_Summary_Results_Trainset =  pd.concat([df_Summary_Results_Trainset, SummaryResultsThisIteration])


        y_pred=best_clf.predict(X_test) # predict the test data

        explained_variance_score = -metrics.explained_variance_score(y_test, y_pred)
        neg_mean_absolute_error_score = metrics.mean_absolute_error(y_test, y_pred)
        neg_mean_squared_error_score = metrics.mean_squared_error(y_test, y_pred, squared=True)
        neg_root_mean_squared_error_score = metrics.mean_squared_error(y_test, y_pred, squared=False)
        r2 = -metrics.r2_score(y_test, y_pred)
        neg_mean_absolute_percentage_error_score = metrics.mean_absolute_percentage_error(y_test, y_pred)

        d = {   'day' : day,
                'model': model['label'], 
                'variance' : [explained_variance_score],
                'mean absolute error' : [neg_mean_absolute_error_score],
                'mean squared error' : [neg_mean_squared_error_score],
                'root mean squared error' : [neg_root_mean_squared_error_score],
                'R2' : [r2],
                'mean absolute percentage error' : [neg_mean_absolute_percentage_error_score],
                'best estimator' : [clf.best_params_],
            }

        SummaryResultsThisIteration =  pd.DataFrame(data=d) 
        df_Summary_Results_Testset =  pd.concat([df_Summary_Results_Testset, SummaryResultsThisIteration])

        #do some stuff
        stop = time.time()
        duration = stop-start
        print("Duration for estimation: {} ".format(duration))



# %% Print Results
writer = pd.ExcelWriter('summaryResults.xlsx')   

print(df_Summary_Results_Trainset)
df_Summary_Results_Trainset.to_excel(writer,
             sheet_name='Trainset') 

print(df_Summary_Results_Testset)
df_Summary_Results_Testset.to_excel(writer,
             sheet_name='Testset') 
writer.save()

# %% plot results
df_Summary_Results_Trainset_groupedByDay = df_Summary_Results_Trainset.groupby("model")
df_Summary_Results_Testset_groupedByDay = df_Summary_Results_Testset.groupby("model")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(26, 12))

#plt.style.use('seaborn-colorblind')
plt.rc('xtick',labelsize='xx-large')
plt.rc('ytick',labelsize='xx-large')

# train 
lines = ['-', '--', '-.', ':']
linecycler = cycle(lines)
for key, modelResult in df_Summary_Results_Trainset_groupedByDay:
    if(key != "Linear Regression"):
        ax1.plot(modelResult["day"], modelResult["root mean squared error"]*100, label='%s' % (key), linewidth=5.0, linestyle=next(linecycler))

# test 
lines = ['-', '--', '-.', ':']
linecycler = cycle(lines)
for key, modelResult in df_Summary_Results_Testset_groupedByDay:
    if(key != "Linear Regression"):
        ax2.plot(modelResult["day"], modelResult["root mean squared error"]*100, label='%s' % (key), linewidth=5.0, linestyle=next(linecycler))


ax1.axis(xmin=1, xmax=28)
ax1.axis(ymin=0.0)
ax1.set_xlabel('Days in the Future', fontsize='xx-large')
ax1.set_ylabel('Root Mean Squared Error (in %)', fontsize='xx-large')
ax1.set_title('PANEL A: Performance on Train Set', fontsize='xx-large')
# axs[0, 0].set_title('Axis [0, 0]')
# axs[0, 0].set_title('Axis [0, 0]')

leg = ax1.legend(loc="upper left", fontsize='xx-large')
# get the lines and texts inside legend box
leg_lines = leg.get_lines()
leg_texts = leg.get_texts()
# bulk-set the properties of all lines and texts

# ax.set_xticklabels(x_ticks, rotation=0, fontsize='x-large')
# ax.set_yticklabels(y_ticks, rotation=0, fontsize='x-large')

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(True)
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()
ax1.tick_params(axis='x', direction='out')
ax1.tick_params(axis='y', length=0)
# offset the spines
for spine in ax1.spines.values():
  spine.set_position(('outward', 5))
# put the grid behind
ax1.set_axisbelow(True)


############## Test set

ax2.axis(xmin=1, xmax=28)
ax2.axis(ymin=0.0)
ax2.set_xlabel('Days in the Future', fontsize='xx-large')
ax2.set_ylabel('Root Mean Squared Error (in %)', fontsize='xx-large')
ax2.set_title('PANEL B: Performance on Test Set', fontsize='xx-large')
# axs[0, 0].set_title('Axis [0, 0]')
# axs[0, 0].set_title('Axis [0, 0]')

leg = ax2.legend(loc="upper left", fontsize='xx-large')
# get the lines and texts inside legend box
leg_lines = leg.get_lines()
leg_texts = leg.get_texts()

# ax.set_xticklabels(x_ticks, rotation=0, fontsize='x-large')
# ax.set_yticklabels(y_ticks, rotation=0, fontsize='x-large')

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(True)
ax2.get_xaxis().tick_bottom()
ax2.get_yaxis().tick_left()
ax2.tick_params(axis='x', direction='out')
ax2.tick_params(axis='y', length=0)
# offset the spines
for spine in ax2.spines.values():
  spine.set_position(('outward', 5))
# put the grid behind
ax2.set_axisbelow(True)
plt.margins(0,0)
plt.savefig("RMSE.pdf", bbox_inches = 'tight', pad_inches = 0)
plt.show()   # Display
# %%
