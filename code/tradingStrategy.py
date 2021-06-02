# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import time
import warnings
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)
np.seterr(divide='ignore')
from scipy import stats
# from statsmodels.tsa.filters.filtertools import recursive_filter

from numba import jit
from numba import njit

os.chdir("C:/Users/rene/Dropbox/rene/cs229/project")

rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage{fourier}")

launch_time = time.time()

Trading_Results_Testset = []

models=["Elastic Net", "Decision Tree Regressor", "XGBoost", "Light GBM"]
thresholds = [0., .0025, .005, .01]

###############################################################################

'#								LOAD DATA'

###############################################################################

for model in models:
    for day in [1, 5, 10, 15, 20, 25]:
        threshold_i = []
        profit_i = []
        trades_i = []
        for threshold in thresholds:

            test_results = pd.read_parquet("results/Summary_Results_Testset.parquet")

            test_results = test_results[test_results["model"] == model].sort_values(by=["date"])
            test_results=test_results[test_results["day"] == day].sort_values(by=["date"])
            yhat = test_results["yhat"].reset_index(drop=True)

            start_= test_results["date"].iat[0]
            EndDate= test_results["date"].iat[-1]
            data = pd.read_parquet("data/databaseFrankfurt.parquet")
            data = data[data["Ticker"]=="VOW.F"]
            #data = data[data.index in test_results["predicted_date"].values].sort_index()
            data = data[data.index.isin(test_results["predicted_date"].values)].sort_index() 
            # shift the values so we get the price of the stock when we buy
            VW=data["Close"].reset_index(drop=True).shift(-day)


            T = len(VW)-day

            def buy_VW(ind, tr, hd):
                return int(1), abs(1.)

            def sell_VW(ind, tr, hd):
                return 0, 0.

            ############################################################################

            #'#                              VARIABLES'

            ############################################################################

            have_VW_l = 0.
            when_bought_l = 0.
            tradeIDs_l = 0

            have_VW_s = 0.
            when_bought_s = 0.
            tradeIDs_s = 0

            c0 =1000. # start with one stock
            a0 = c0
            m0 = c0
            q0 = 0.
            number_trades = 0

            shift = 1

            C = c0*np.ones(T-shift+1)
            A = a0*np.ones(T-shift+1)
            M = m0*np.ones(T-shift+1)

            BL = np.zeros(T-shift+1)
            SL = np.zeros(T-shift+1)

            BS = np.zeros(T-shift+1)
            SS = np.zeros(T-shift+1)

            SL_l = np.zeros(T-shift+1)
            SL_s = np.zeros(T-shift+1)

            ThresholdLong = threshold
            ThresholdShort = threshold

            counter = 0.
            qt = 0.

            cmax = c0

            for pt in range(T):

                # read VW
                VW_a = VW[pt]
                VW_b = VW[pt]

                cmax = max(c0, cmax)

                qt += 1.
                
                # buying

                if have_VW_l == 0. and yhat[pt]>ThresholdLong:
                    tradeIDs_l, have_VW_l = buy_VW(
                        'L', tradeIDs_l, have_VW_l)
                    when_bought_l = VW_a
                    BL[pt-shift+1] = VW_a
                    number_trades += 1

                if have_VW_s == 0. and yhat[pt]<ThresholdShort:
                    tradeIDs_s, have_VW_s = buy_VW(
                        'S', tradeIDs_s, have_VW_s)
                    when_bought_s = VW_b
                    BS[pt-shift+1] = VW_b
                    number_trades += 1

                # selling

                if have_VW_l != 0. and yhat[pt]<ThresholdLong:
                    c0 += have_VW_l*(VW_b-when_bought_l)
                    tradeIDs_l, have_VW_l = sell_VW('L', tradeIDs_l, have_VW_l)
                    when_bought_l = 0.
                    SL[pt-shift+1] = VW_b
                    number_trades += 1

                if have_VW_s != 0. and yhat[pt]>ThresholdShort:
                    c0 += have_VW_s*(when_bought_s-VW_a)
                    tradeIDs_s, have_VW_s = sell_VW('S', tradeIDs_s, have_VW_s)
                    when_bought_s = 0.
                    SS[pt-shift+1] = VW_a
                    number_trades += 1

                a0 = c0+have_VW_l*(VW_b-when_bought_l) + \
                    have_VW_s*(when_bought_s-VW_a)
                m0 = a0-0.01*0.5*(VW_a+VW_b)*(have_VW_s+have_VW_l)

                C[pt-shift+1] = c0
                A[pt-shift+1] = a0
                M[pt-shift+1] = m0


                if M[pt-shift+1] <= 0. or A[pt-shift+1] <= 0.:
                    print("broke")
                    break

                # if qt >= T/100.:
                #     qt = 0.
                #     print(round(a0, 2), round((a0/A[0]-1.)*100., 2), round((VW[pt]/VW[0]-1.)*100., 2), round(float(pt)/float(T-1)*100., 2))

                profit = (A[T-shift]/A[0]-1.)*100.

            # Save result
            threshold_i.append(threshold)
            profit_i.append(profit)
            trades_i.append(number_trades)

        d = {   'model': model, 
        'day' : day, # hstep
        'profit_threshold1' : profit_i[0],
        'profit_threshold2' : profit_i[1],
        'profit_threshold3' : profit_i[2],
        'profit_threshold4' : profit_i[3],
        'trades_threshold1' : trades_i[0],
        'trades_threshold2' : trades_i[1],
        'trades_threshold3' : trades_i[2],
        'trades_threshold4' : trades_i[3],
            }            
        Trading_Results_Testset.append(d)

df_Trading_Results_Testset =  pd.DataFrame(Trading_Results_Testset)  
df_Trading_Results_Testset.to_parquet("results/Trading_Results_Testset.parquet", engine = "fastparquet", compression = "gzip")

writer = pd.ExcelWriter('results/Trading_Results_Testset.xlsx')   

print(df_Trading_Results_Testset)
df_Trading_Results_Testset.to_excel(writer,
             sheet_name='Testset') 
writer.save()
# %% In case someone wants to see a plot of all longs and shorts 
#  as well as the overall portfolio trajectory
t = np.linspace(0., float(T), num=T-shift+1, endpoint='true')

rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage{fourier}")

rc('font', size=20)


plt.plot(t, C, color='b', label="Capital")
plt.plot(t, A, color='g', label="Account")
plt.plot(t, M, color='r', label="Margin")

MC = np.max(C)


rc('font', size=20)

plt.show()

# %%
