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

os.chdir("C:/Users/Rene/Dropbox/rene/cs229/project")

'''
//......................................................................................................................................................
//.PPPPPPPPPPPPP................................................................................... DDDDDDDDDDDD....................ttt.................
//.PPPPPPPPPPPPPP.................................................................................. DDDDDDDDDDDDD.................ttttt.................
//.PPPPPPPPPPPPPP.................................................................................. DDDDDDDDDDDDDD................ttttt.................
//.PPPPP..PPPPPPPP................................................................................. DDDD..DDDDDDDD................ttttt.................
//.PPPPP....PPPPPP.Prrrrrrrr..eeeeeeee....ppppppppppp.....aaaaaaaa....rrrrrrrrr..eeeeeeee.......... DDDD.....DDDDDD...aaaaaaaaa.aattttttt..aaaaaaaaa....
//.PPPPP.....PPPPP.Prrrrrrrr.eeeeeeeeee...pppppppppppp...aaaaaaaaaa...rrrrrrrrrreeeeeeeeee......... DDDD.....DDDDDD..aaaaaaaaaa.aattttttt.aaaaaaaaaaa...
//.PPPPP....PPPPPP.Prrrrrrrrreeeeeeeeeee..ppppppppppppp.paaaaaaaaaaa..rrrrrrrrrreeeeeeeeee......... DDDD......DDDDD.Daaaaaaaaaaaaattttttt.aaaaaaaaaaa...
//.PPPPP..PPPPPPPP.Prrrrr...reeeeeeeeeee..ppppppppppppp.paaaa..aaaaa..rrrrrr..rreeee.eeeeee........ DDDD......DDDDD.Daaaa..aaaaa..ttttt..taaaa..aaaaa...
//.PPPPPPPPPPPPPP..Prrrr...rreee...eeeee..ppppp...ppppp.......aaaaaa..rrrrr...rreee...eeeee........ DDDD......DDDDD........aaaaa..ttttt.........aaaaa...
//.PPPPPPPPPPPPP...Prrrr...rreeeeeeeeeee..ppppp...ppppp...aaaaaaaaaa..rrrrr...rreeeeeeeeeee........ DDDD......DDDDD...aaaaaaaaaa..ttttt.....aaaaaaaaa...
//.PPPPPPPPPPPP....Prrrr...rreeeeeeeeeee..pppp....ppppp.paaaaaaaaaaa..rrrrr...rreeeeeeeeeee........ DDDD......DDDDD.Daaaaaaaaaaa..ttttt...aaaaaaaaaaa...
//.PPPPP...........Prrrr...rreee..........ppppp...ppppp.paaaaaaaaaaa..rrrr....rreee................ DDDD.....DDDDDD.Daaaaaaaaaaa..ttttt..taaaaaaaaaaa...
//.PPPPP...........Prrrr...rreeee.........ppppp...pppppppaaaa..aaaaa..rrrr....rreee................ DDDD.....DDDDDD.Daaaa..aaaaa..ttttt..taaaa..aaaaa...
//.PPPPP...........Prrrr....reeeeeeeeeee..pppppppppppppppaaa..aaaaaa..rrrr....rreeeeeeeeeee........ DDDD...DDDDDDD..Daaa...aaaaa..ttttttttaaaa..aaaaa...
//.PPPPP...........Prrrr....reeeeeeeeeee..pppppppppppppppaaaaaaaaaaa..rrrr.....reeeeeeeeee......... DDDDDDDDDDDDDD..Daaaaaaaaaaa..ttttttttaaaaaaaaaaa...
//.PPPPP...........Prrrr.....eeeeeeeeee...pppppppppppp..paaaaaaaaaaa..rrrr.....reeeeeeeeee......... DDDDDDDDDDDDD...Daaaaaaaaaaa..ttttttttaaaaaaaaaaa...
//.PPPPP...........Prrrr......eeeeeeee....ppppppppppp....aaaaaaaaaaa..rrrr.......eeeeeeee.......... DDDDDDDDDDD......aaaaaaaaaaaa..tttttttaaaaaaaaaaaa..
//........................................ppppp.........................................................................................................
//........................................ppppp.........................................................................................................
//........................................ppppp.........................................................................................................
//........................................ppppp.........................................................................................................
//........................................ppppp.........................................................................................................
//......................................................................................................................................................


'''
# %% Read Data
data = pd.read_parquet("data/databaseMergedFrankfurt.parquet")

tickerList = pd.read_excel("data/VWSupplyChain.xlsx", sheet_name="TickerFrankfurt")
tickers = tickerList["Yahoo Finance Ticker"].dropna()
# %% First Only work with next days Return
y = data["Return_Future_Delta1_VW"]
trainColumns = ["Return_Future_Delta1_VW"]
for j, ticker in enumerate(tickers):
    for i in range(1,28):
        trainColumns.append("Return_Delta{}_{}".format(i, ticker))

for i in range(1,28):
    trainColumns.append("Return_Delta{}_VW".format(i))
    
X = data[trainColumns]

print(X.head())
# %%
print(X.info())
# %%
print(X.describe())




# %%
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)

from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split

trainSet, testSet = train_test_split(X, train_size=.8, random_state=42)
#tscv = TimeSeriesSplit()
#tscv = TimeSeriesSplit(X, n_splits=int((len(y)-3)/3))

# for train_index, test_index in tscv.split(X):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]

y_train = trainSet["Return_Future_Delta1_VW"].values
y_test = testSet["Return_Future_Delta1_VW"].values

X_train = trainSet.drop(["Return_Future_Delta1_VW"], axis=1)
X_test = testSet.drop(["Return_Future_Delta1_VW"], axis=1)

# %%

from sklearn.linear_model import LinearRegression

estimator = LinearRegression()
estimator.fit(X_train, y_train)


# %%
'''
//......................................................................................................................................................
//.FFFFFFFFFFFFFFFFF..iiiii.....................MMMMMMMM.......MMMMMMMM...............................ddddd....................lllll....................
//.FFFFFFFFFFFFFFFFF..iiiii......ttt............MMMMMMMMM.....MMMMMMMMM...............................ddddd....................lllll....................
//.FFFFFFFFFFFFFFFFF..iiiii....ttttt............MMMMMMMMM.....MMMMMMMMM...............................ddddd....................lllll....................
//.FFFFFFFFFFFFFFFFF...........ttttt............MMMMMMMMM.....MMMMMMMMM...............................ddddd....................lllll....................
//.FFFFFF......................ttttt............MMMMMMMMMM....MMMMMMMMM...............................ddddd....................lllll....................
//.FFFFFF......................ttttt............MMMMMMMMMM...MMMMMMMMMM...............................ddddd....................lllll....................
//.FFFFFF.............iiiii..ittttttttt.........MMMMMMMMMM...MMMMMMMMMM.....oooooooooo........ddddddddddddd......eeeeeeeee.....lllll....sssssssssss.....
//.FFFFFF.............iiiii..ittttttttt.........MMMMMMMMMM...MMMMMMMMMM....oooooooooooo......dddddddddddddd....eeeeeeeeeeee....lllll...sssssssssssss....
//.FFFFFF.............iiiii..ittttttttt.........MMMMMMMMMMM..MMMMMMMMMM...oooooooooooooo....ddddddddddddddd...eeeeeeeeeeeeee...lllll...ssssssssssssss...
//.FFFFFFFFFFFFFFF....iiiii....ttttt............MMMMMMMMMMM.MMMMMMMMMMM..oooooooooooooooo..oddddddddddddddd...eeeeeeeeeeeeee...lllll..lssssss.sssssss...
//.FFFFFFFFFFFFFFF....iiiii....ttttt............MMMMMMMMMMM.MMMMMMMMMMM..oooooo....ooooooo.oddddd....dddddd..deeeee....eeeeee..lllll..lsssss...ssssss...
//.FFFFFFFFFFFFFFF....iiiii....ttttt............MMMMMMMMMMM.MMMMMMMMMMM.Moooooo.....oooooo.oddddd....dddddd..deeeee....eeeeee..lllll..lsssssss..........
//.FFFFFFFFFFFFFFF....iiiii....ttttt............MMMMMM.MMMMMMMMMMMMMMMM.Mooooo......oooooo.odddd......ddddd..deeeeeeeeeeeeeee..lllll..lsssssssssss......
//.FFFFFF.............iiiii....ttttt............MMMMMM.MMMMMMMMM.MMMMMM.Mooooo......oooooo.odddd......ddddd..deeeeeeeeeeeeeee..lllll...sssssssssssss....
//.FFFFFF.............iiiii....ttttt............MMMMMM.MMMMMMMMM.MMMMMM.Mooooo......oooooo.odddd......ddddd..deeeeeeeeeeeeeee..lllll....sssssssssssss...
//.FFFFFF.............iiiii....ttttt............MMMMMM.MMMMMMMMM.MMMMMM.Mooooo......oooooo.odddd......ddddd..deeeee............lllll......ssssssssssss..
//.FFFFFF.............iiiii....ttttt............MMMMMM.MMMMMMMMM.MMMMMM.Moooooo.....oooooo.oddddd....dddddd..deeeee............lllll..........ssssssss..
//.FFFFFF.............iiiii....ttttt............MMMMMM..MMMMMMM..MMMMMM..oooooo....oooooo..oddddd....dddddd..deeeeee...........lllll..lsssss....ssssss..
//.FFFFFF.............iiiii....tttttttt.........MMMMMM..MMMMMMM..MMMMMM..oooooooooooooooo..oddddddddddddddd...eeeeeee.eeeeeee..lllll..lssssss.ssssssss..
//.FFFFFF.............iiiii....tttttttt.........MMMMMM..MMMMMMM..MMMMMM...oooooooooooooo....ddddddddddddddd...eeeeeeeeeeeeeee..lllll..lssssssssssssss...
//.FFFFFF.............iiiii....ttttttttt........MMMMMM..MMMMMMM..MMMMMM....oooooooooooo......dddddddddddddd....eeeeeeeeeeeee...lllll...ssssssssssssss...
//.FFFFFF.............iiiii.....tttttttt........MMMMMM...MMMMM...MMMMMM.....oooooooooo........ddddddddddddd......eeeeeeeeee....lllll....sssssssssss.....
//......................................................................................................................................................


'''

from sklearn.linear_model import LinearRegression
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


# Add the models to the list 
models = [
{
    'label': 'Linear Regression',
    'estimator': LinearRegression(),
    'param_grid' : {
        #'C': np.logspace(-10, 3, 5)
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
        # 'reg_alpha': [0, 1e-2, 0.5, 1, 1e1],
        # 'reg_lambda': [0, 1e-2, 0.5, 1, 1e1],
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
        # 'reg_alpha': [0, 1e-2, 0.5, 1, 1e1],
        # 'reg_lambda': [0, 1e-2, 0.5, 1, 1e1],
        'num_leaves': [11, 31, 61, 101],
        # 'min_child_weight': [0, .01, 0.5, 1],
        # 'min_split_gain': [0, 0.1, 0.2],
        # 'min_data_in_leaf': [30, 50, 100, 300, 400],
    },
    'linestyle' : '-.', # (0, (3, 5, 1, 5, 1, 5)),
}
]


df_Summary_Results_Trainset = pd.DataFrame()
df_Summary_Results_Testset = pd.DataFrame()

# Below for loop iterates through your models list
for j, model in enumerate(models):
    
    print(model['label'])
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

    d = {'model': model['label'], 
                'explained variance' : [explained_variance_score],
                'mean absolute error' : [neg_mean_absolute_error_score],
                'mean squared error' : [neg_mean_squared_error_score],
                'root mean squared error' : [neg_root_mean_squared_error_score],
                'R2' : [r2],
                'mean absolute percentage error' : [neg_mean_absolute_percentage_error_score]
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

    d = {'model': model['label'], 
                'variance' : [explained_variance_score],
                'mean absolute error' : [neg_mean_absolute_error_score],
                'mean squared error' : [neg_mean_squared_error_score],
                'root mean squared error' : [neg_root_mean_squared_error_score],
                'R2' : [r2],
                'mean absolute percentage error' : [neg_mean_absolute_percentage_error_score]
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