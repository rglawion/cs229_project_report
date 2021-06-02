# %%
import pandas as pd
import numpy as np
from scipy.stats import lognorm
import datetime as dt
import os
import seaborn as sns
from numba import jit
import matplotlib.pyplot as plt

import joblib
import tqdm

import math 
import copy

import time
from dateutil.relativedelta import *
from datetime import datetime

import time

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lightgbm

#from sklearn.model_selection import GridSearchCV
from hypopt import GridSearch    # better for fixed validation set

from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import ParameterGrid
from numpy.core.numeric import Inf

from sklearn import metrics

from itertools import cycle

import warnings
warnings.filterwarnings("ignore")

def GridSearchReneParallelHelper(X_train, y_train, X_val, y_val, estimator, params):
    est = estimator.set_params(**params)
    est.fit(X_train, y_train)
    y_pred=est.predict(X_val)
    score = mean_squared_error(y_val, y_pred, squared=True)
    d = {'params': [params],
         'score' : score}

    result_i = pd.DataFrame(data=d) 
    return result_i

def GridSearchRene(X_train, y_train, X_val, y_val, estimator, parameters, parallel=False):
    '''
    Returns the best parameter combination on the validation set

    Inputs: Training Set
            Validation Set
            Estimator to test
            Parameters :    Parameter grid to test

    Returns:
            Best parameters
    '''
    grid = list(ParameterGrid(parameters))

    if parallel:
        scores = joblib.Parallel( n_jobs = 42 )( joblib.delayed(GridSearchReneParallelHelper)(X_train, y_train, X_val, y_val, estimator, params,)
            for params in grid)
        scores = pd.concat(scores, axis=0)
        scores.sort_values(by=["score"], ascending=True, inplace=True)
        best_params = scores["params"].iat[0]
    else:
        best_score = Inf
        best_params = []
        for params in grid:
            est = estimator.set_params(**params)
            est.fit(X_train, y_train)
            y_pred=est.predict(X_val)
            score = mean_squared_error(y_val, y_pred, squared=True)
            if score < best_score:
                best_score = score
                best_params = params

    return best_params

os.chdir("C:/Users/rene/Dropbox/rene/cs229/project")

# %%
Summary_Results_Trainset = []
Summary_Results_Valset = []
Summary_Results_Testset = []

EndOfTrainDate2010 = datetime(2010,1,1)

# %% Read Data
#data = pd.read_parquet("data/databaseMergedFrankfurt.parquet")
data = pd.read_parquet("data/databaseFrankfurtComplete.parquet")
data["Date"] = pd.to_datetime(data["Date"].dt.date)
data.index = data["Date"]

tickerList = pd.read_excel("data/VWSupplyChain.xlsx", sheet_name="TickerFrankfurt")
tickers = tickerList["Yahoo Finance Ticker"].dropna()

# %%
###################
####### Models we want to use
###################
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
        'min_samples_split': [2, 4, 10, 20],
        'min_samples_leaf' : [2, 4, 8],
        'max_features' : ['auto', 'sqrt', 'log2']
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

####################
####    Always use those columns for prediction
#####################
trainColumnsfixed = []  
for j, ticker in enumerate(tickers):
    for i in range(1,28):
        trainColumnsfixed.append("Return_Delta{}_{}".format(i, ticker))
    
for i in range(1,28):
    trainColumnsfixed.append("Return_Delta{}_VW".format(i))

####################
####    Always use macro variables
#####################
macro_variables = ["euribor_3m" ,"eurirs_10y", 
                        "itraxx_crossover1", 
                        "itraxx_crossover1_relDiff1", "itraxx_crossover1_relDiff6",
                        "euribor_3m_relDiff1", "eurirs_10y_relDiff1", 
                        "euribor_3m_relDiff6", "eurirs_10y_relDiff6",
                        "OutputGap", "CPI_x", "bund_y10_x", "ISwap_y10_x"]
for macro_variable in macro_variables:
    trainColumnsfixed.append(macro_variable)

months_to_iterate_over = math.floor(((data["Date"].iat[-1] - EndOfTrainDate2010)/np.timedelta64(1, 'M')))

print("Total months to iterate over: {}".format(months_to_iterate_over))

# %% Main Loop
for month in range(months_to_iterate_over-10):

    print("We are in month: {} of {}.".format(month+1, months_to_iterate_over-10))

    for day in [1, 5, 10, 15, 20, 25]:
    

        ####################
        ####    Select Data we want to train on
        #####################
        trainColumns = copy.deepcopy(trainColumnsfixed)
        trainColumns.append("Return_Future_Delta{}_VW".format(day))

        X = data[trainColumns]

        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)

        # Drop columns if stock does not exist
        # X.dropna(axis=1, how='any', inplace=True)

        ####################
        ####    Get Dates for each training period
        #####################
        trainDates = EndOfTrainDate2010 +relativedelta(months=month, day=31, weekday=FR(-1))
        validationDates =  trainDates+relativedelta(months=12, day=31, weekday=FR(-1))
        testDates =  validationDates+relativedelta(months=1, day=31, weekday=FR(-1))

        ####################
        ####    Select training, validation, and test samples
        #####################

        trainSet = X[X.index<trainDates]
        valSet = X[(X.index>trainDates) & (X.index<validationDates)]
        testSet = X[(X.index>validationDates) & (X.index<testDates)]

        y_train = trainSet["Return_Future_Delta{}_VW".format(day)].values
        y_val = valSet["Return_Future_Delta{}_VW".format(day)].values
        y_test = testSet["Return_Future_Delta{}_VW".format(day)].values

        X_train = trainSet.drop(["Return_Future_Delta{}_VW".format(day)], axis=1)
        X_val = valSet.drop(["Return_Future_Delta{}_VW".format(day)], axis=1)
        X_test = testSet.drop(["Return_Future_Delta{}_VW".format(day)], axis=1)


        ####################
        ####   Below for loop iterates through your models list
        #####################

        for j, model in enumerate(models):
            
            print("Day : {} \t model : {}".format(str(day), model['label']))
            start = time.time()
            
            estimator = model['estimator'] # select the model
            parameters = model['param_grid']

            ####################
            ####  Perform our Gridsearch to select the best model
            #####################
            best_params = GridSearchRene(X_train=X_train, y_train=y_train, 
                                            X_val=X_val, y_val=y_val,
                                            estimator=estimator, parameters=parameters,
                                            parallel=True)

            best_regressor = estimator
            best_regressor.set_params(**best_params)
            best_regressor.fit(X_train, y_train)

            y_pred=best_regressor.predict(X_train) # predict the train data

            # Save time for prediction
            stop = time.time()
            duration = stop-start
            print("Duration for estimation: {} ".format(duration))

            ####################
            ####  Save Results Training set
            #####################
            explained_variance_score = -metrics.explained_variance_score(y_train, y_pred)
            neg_mean_absolute_error_score = metrics.mean_absolute_error(y_train, y_pred)
            neg_mean_squared_error_score = metrics.mean_squared_error(y_train, y_pred, squared=True)
            neg_root_mean_squared_error_score = metrics.mean_squared_error(y_train, y_pred, squared=False)
            r2 = -metrics.r2_score(y_train, y_pred)
            neg_mean_absolute_percentage_error_score = metrics.mean_absolute_percentage_error(y_train, y_pred)

            for ii, prediction in enumerate(y_pred):
                d = {   'model': model['label'], 
                        'day' : day, # hstep
                        'explained_variance' : explained_variance_score,
                        'mean absolute_error' :neg_mean_absolute_error_score,
                        'mean_squared_error' : neg_mean_squared_error_score,
                        'root_mean_squared_error' : neg_root_mean_squared_error_score,
                        'R2' : r2,
                        'mean_absolute_percentage_error' : neg_mean_absolute_percentage_error_score,
                        'best_estimator' : [best_params],
                        'estimation_time_complete' : duration,
                        'date' : X_val.index[-1],
                        'predicted_date' : X_train.index[ii],
                        'y' : y_train[ii],
                        'yhat' : prediction,
                        'err' : y_train[ii] - prediction,
                    }
                
                Summary_Results_Trainset.append(d)

            ####################
            ####  Save Results Validation set
            #####################

            y_pred=best_regressor.predict(X_val) # predict the validation data

            explained_variance_score = -metrics.explained_variance_score(y_val, y_pred)
            neg_mean_absolute_error_score = metrics.mean_absolute_error(y_val, y_pred)
            neg_mean_squared_error_score = metrics.mean_squared_error(y_val, y_pred, squared=True)
            neg_root_mean_squared_error_score = metrics.mean_squared_error(y_val, y_pred, squared=False)
            r2 = -metrics.r2_score(y_val, y_pred)
            neg_mean_absolute_percentage_error_score = metrics.mean_absolute_percentage_error(y_val, y_pred)

            for ii, prediction in enumerate(y_pred):
                d = {   'model': model['label'], 
                        'day' : day, # hstep
                        'explained_variance' : explained_variance_score,
                        'mean absolute_error' :neg_mean_absolute_error_score,
                        'mean_squared_error' : neg_mean_squared_error_score,
                        'root_mean_squared_error' : neg_root_mean_squared_error_score,
                        'R2' : r2,
                        'mean_absolute_percentage_error' : neg_mean_absolute_percentage_error_score,
                        'best_estimator' : [best_params],
                        'estimation_time_complete' : duration,
                        'date' : X_val.index[-1],
                        'predicted_date' : X_val.index[ii],
                        'y' : y_val[ii],
                        'yhat' : prediction,
                        'err' : y_val[ii] - prediction,
                    }
                
                Summary_Results_Valset.append(d)

            ####################
            ####     Save Results test set
            #####################

            y_pred=best_regressor.predict(X_test) # predict the test data

            explained_variance_score = -metrics.explained_variance_score(y_test, y_pred)
            neg_mean_absolute_error_score = metrics.mean_absolute_error(y_test, y_pred)
            neg_mean_squared_error_score = metrics.mean_squared_error(y_test, y_pred, squared=True)
            neg_root_mean_squared_error_score = metrics.mean_squared_error(y_test, y_pred, squared=False)
            r2 = -metrics.r2_score(y_test, y_pred)
            neg_mean_absolute_percentage_error_score = metrics.mean_absolute_percentage_error(y_test, y_pred)

            for ii, prediction in enumerate(y_pred):
                d = {   'model': model['label'], 
                        'day' : day, # hstep
                        'explained_variance' : explained_variance_score,
                        'mean absolute_error' :neg_mean_absolute_error_score,
                        'mean_squared_error' : neg_mean_squared_error_score,
                        'root_mean_squared_error' : neg_root_mean_squared_error_score,
                        'R2' : r2,
                        'mean_absolute_percentage_error' : neg_mean_absolute_percentage_error_score,
                        'best_estimator' : [best_params],
                        'estimation_time_complete' : duration,
                        'date' : X_val.index[-1],
                        'predicted_date' : X_test.index[ii],
                        'y' : y_test[ii],
                        'yhat' : prediction,
                        'err' : y_val[ii] - prediction,
                    }
                
                Summary_Results_Testset.append(d)



# %% Print and Save Results
df_Summary_Results_Trainset =  pd.DataFrame(Summary_Results_Trainset)    
df_Summary_Results_Valset = pd.DataFrame(Summary_Results_Valset)    
df_Summary_Results_Testset = pd.DataFrame(Summary_Results_Testset)    

# Save to database for better usage for later analysis
df_Summary_Results_Trainset.to_parquet("results/Summary_Results_Trainset.parquet", engine = "fastparquet", compression = "gzip")
df_Summary_Results_Valset.to_parquet("results/Summary_Results_Valset.parquet", engine = "fastparquet", compression = "gzip")
df_Summary_Results_Testset.to_parquet("results/Summary_Results_Testset.parquet", engine = "fastparquet", compression = "gzip")

# save to excel for summary - too large for excel
# writer = pd.ExcelWriter('results/summaryResults.xlsx')   

# print(df_Summary_Results_Trainset)
# df_Summary_Results_Trainset.to_excel(writer,
#              sheet_name='Trainset') 

# print(df_Summary_Results_Valset)
# df_Summary_Results_Valset.to_excel(writer,
#              sheet_name='Valdiationset') 

# print(df_Summary_Results_Testset)
# df_Summary_Results_Testset.to_excel(writer,
#              sheet_name='Testset') 
# writer.save()

# %%
