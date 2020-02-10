import pandas as pd
import numpy as np
import os
os.chdir("/mnt/data/Google Drive/application quant trading/python")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil import parser
# from pandas.tseries import converter
# register_matplotlib_converters()
# import statsmodels.formula.api as smf
# import statsmodels.tsa.api as smt
# import statsmodels.api as sm
# import scipy.stats as scs
# import statsmodels.stats as sms
# import pandas.util.testing as tm
import seaborn as sns
sns.set_style('darkgrid')
import functions as f

## ================================================================== ##
##                Read Data and do Data manipulation                  ##
## ================================================================== ##

class backtest: 
    def __init__(self, df = None, file_id = "", strategy = "momentum_trading"):
        if df is None: 
            self.df = f.load_data()
        else: 
            self.df = df

        self.file = file_id
        self.df_rt = f.make_returns(self.df)
        self.df_lrt = f.make_log_returns(self.df)
        self.df_lrt_30ma = f.get_moving_average(self.df_lrt)
        self.indices = self.df_lrt.index
        self.n = len(self.df_lrt)
        self.tradecosts = 0.001
        self.strategy = strategy
        self.assets = self.df.columns
        self.endowment = np.random.choice(self.assets, 5, replace = False)
        self.startperiod = 31
        self.cur_date_ind = self.startperiod
        self.start_date = self.indices[self.cur_date_ind] 
        self.current_date = self.start_date
        self.cur_return = 0
        self.baseline_return = 0
        self.tradefreq = 14
        self.cash = 0
        self.cash_baseline = 0
        self.finish = False

        self.endowment = f.get_worst_performers(self.df_lrt_30ma, self.current_date)
        self.cash -= np.sum(self.df.loc[self.current_date, self.endowment] * (1 + self.tradecosts))
        self.cash_baseline -= np.sum(self.df.loc[self.current_date, :]) * (1 + self.tradecosts)

        ## current value when cashing out
        self.cur_value = self.cash + np.sum(self.df.loc[self.current_date, self.endowment]) * (1 - self.tradecosts)
        self.cur_value_baseline = self.cash_baseline + np.sum(self.df.loc[self.current_date, :]) * (1 - self.tradecosts)
        self.overall_return = 1
        self.overall_return_base = 1
        self.results = []
        self.threshold = None

    def increment_time(self, days = None):
        if days is None:
            days = self.tradefreq
        
        end_date = self.indices[min(self.cur_date_ind + days, (self.n -2))]

        while (self.current_date <= end_date):
            self.cur_date_ind += 1
            self.current_date = self.indices[self.cur_date_ind]
            
            # Pseudocode implementation of stoploss
            # if any stock in endowment < self.threshold:
            #    trade(self, strategy == "stoploss", stoploss = that stock)
            
        if self.current_date == self.indices[-1]:
            self.finish = True


    def trade(self, strategy = None, stoploss = None):
        if strategy is None: 
            strategy = self.strategy
        if strategy == "mean_reversion":
            desired_stocks = f.get_worst_performers(self.df_lrt_30ma, self.current_date)

        if strategy == "momentum_trading":
            desired_stocks = f.get_best_performers(self.df_lrt_30ma, self.current_date)

        if strategy == "prediction_based":
            stock_predictions = f.predict_ARMA(self.df_lrt, self.cur_date_ind, days_to_forecast = self.tradefreq)
            desired_stocks = self.assets[stock_predictions > 0.7]

        if strategy == "stoploss":
            desired_stocks = self.endowment.remove(stoploss)

        # buy assets
        for asset in desired_stocks:
            if asset not in self.endowment:
                self.cash -= (1 + self.tradecosts) * self.df.loc[self.current_date, asset]

        # sell assets
        for asset in self.endowment: 
            if asset not in desired_stocks: 
                self.cash += self.df.loc[self.current_date, asset] * (1 - self.tradecosts)

        self.endowment = desired_stocks 

    def track_profits(self): 
        ## simulate selling everything
        # cur value = cash + what I get if I sell everything
        new_cur_value = self.cash + np.sum(self.df.loc[self.current_date, self.endowment]) * (1 - self.tradecosts)

        new_cur_value_baseline = self.cash_baseline + np.sum(self.df.loc[self.current_date, :]) * (1 - self.tradecosts)

        # self.cur_last_return = (new_cur_value / self.cur_value)
        # self.cur_last_return_base = (new_cur_value_baseline / self.cur_value_baseline)

        self.cur_value = new_cur_value
        self.cur_value_baseline = new_cur_value_baseline
        
        # self.overall_return = self.overall_return * self.cur_last_return
        # self.overall_return_base = self.overall_return_base * self.cur_last_return_base

        # self.days_passed = min(1, (self.current_date - self.start_date).days)
        # self.overall_dreturn = self.overall_return **(1.0/self.days_passed)
        # self.overall_dreturn_base = self.overall_return_base **(1.0/(self.days_passed))

        self.results.append([self.current_date, self.cur_value, self.cur_value_baseline])

    def predict_ARMA(self):
        res = f.predict_ARMA(self.df_lrt, self.cur_date_ind, days_to_forecast = self.tradefreq)
        return(res)


    def plot_assets(self):
        f.plot_all(self.df, self.file)

    def plot_assets_log(self):
        f.plot_all_log(self.df, self.file)

    def plot_autocorrelations(self):
        f.plot_autocorrelations(self.df_lrt, self.file)

    def examine_stationarity(self):
        self.stationarity = f.examine_stationarity(self.df_lrt, self.file)

    def examine_cointegration(self):
        self.cointegration = f.examine_cointegration(self.df_lrt, self.file)

    def examine_conditional_heterosk(self):
        df_lrt_square = self.df_lrt.copy() ** 2
        f.plot_autocorrelations(df_lrt_square, file = self.file + "squared")

    def select_arma_order(self):
        self.arma_order = f.select_arma_order(self.df_lrt, self.file)

    def run_analysis(self):
        f.plot_all(self.df, self.file)
        f.plot_all_log(self.df, self.file)
        f.plot_autocorrelations(self.df_lrt, self.file)
        f.examine_stationarity(self.df_lrt, self.file)
        f.examine_cointegration(self.df, self.file)
        f.plot_autocorrelations(self.df_lrt ** 2, file = self.file + "squared")
        f.select_arma_order(self.df_lrt, self.file)

    def examine_strategies(self):
        d = pd.DataFrame(self.results)
        d.columns = ['Date', 'strategy', 'baseline']
        d.index = d['Date']
        d = d.drop(['Date'], axis = 1)
        self.results = d
        fig = plt.figure(figsize=(10, 10))
        fig.suptitle("Results " + self.strategy, fontsize=14)
        plt.plot(d)
        fig.savefig("../results/" + self.file + self.strategy + "_results.jpg")

        d.to_csv("../results/" + self.file + self.strategy + "_results.csv")


df = f.load_data()
df_train = df

#df_train = df.iloc[0:1500, :]
df_train = df_train.drop(['Asset2', 'Asset25'], axis = 1)

# backtest = backtest(df_train, file_id = "train_")
# backtest = backtest(df_train, file_id = "train_", strategy = "mean_reversion")
backtest = backtest(df_train, file_id = "results_full_ts/outlier_removed_", strategy = "mean_reversion")


while (not backtest.finish):
    backtest.trade()
    backtest.increment_time()
    backtest.track_profits()

# backtest.run_analysis()
backtest.examine_strategies()
print(backtest.results)
plt.show()

# backtest.plot_autocorrelations()





#print(backtest.select_arma_order())

#print(backtest.predict_ARMA())


# ======================================================== # 
# ARMA forecast
# ======================================================== #

# tmp = df_lrt['Asset1']

# predictions = []
# predictions_se = []
# predictions_ci = []

# for i in range(5, len(tmp)):
#     try: 
#         model = smt.ARMA(tmp[max(0, i - 30):i], (1,1))
#         model_fit = model.fit()
#     except: 
#         model = smt.ARMA(max(0, i - 30), (0,0))
#         model_fit = model.fit()
#     prediction = model_fit.forecast()
#     predictions.append(prediction[0])
#     predictions_se.append(prediction[1])
#     predictions_ci.append(prediction[2].flatten())

# # format so we can create a DataFrame
# predictions = pd.DataFrame(predictions)
# predictions = pd.DataFrame(np.concatenate((np.repeat(np.nan,4), predictions.values.flatten()), axis = 0))

# predictions_se = pd.DataFrame(predictions_se)
# predictions_se = pd.DataFrame(np.concatenate((np.repeat(np.nan,4), predictions_se.values.flatten()), axis = 0))

# tmp = [np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), np.array([np.nan, np.nan])]
# predictions_ci = pd.DataFrame(tmp + predictions_ci)

# try:
#    forecasts.head()
# except NameError:
#    forecasts = pd.DataFrame()

# forecasts['ARMA11'] = predictions.values.flatten()
# forecasts['ARMA11_s.e.'] = predictions_se.values.flatten()
# forecasts['ARMA11_ci_low'] = predictions_ci[0].values.flatten()
# forecasts['ARMA11_ci_up'] = predictions_ci[1].values.flatten()

# ARMA11_mse = np.sum((forecasts['ARMA11'].values[4:] - V.values[4:]) ** 2)
# print('MSE')
# print(ARMA11_mse)

# # accuracy
# forecasts['eins'] = 1
# print('Accuracy')
# print(np.sum(np.sign(V[4:]) == np.sign(forecasts['ARMA11'][4:].values)) / len(V[4:]))
# print('Accuracy always up')
# print(np.sum(np.sign(V[4:]) == np.sign(forecasts['eins'][4:].values)) / len(V[4:]))

# fig = plt.figure()
# y = forecasts['ARMA11'][4:]
# z = V[4:].values
# x = np.linspace(0, 1505, 1505)
# plt.fill_between(x, forecasts['ARMA11_ci_low'][4:], forecasts['ARMA11_ci_up'][4:], color = 'r', alpha = 0.4)
# plt.plot(x, z)
# plt.plot(x, y, color = "orange")
# fig.savefig('Asset1_ARMA11_forecast', bbox_inches='tight')










# from statsmodels.tsa.stattools import coint

# cointegr = []
# for asset1 in assets:
#     for asset2 in assets:
#         if asset1 != asset2:
#             x = df[asset1]
#             y = df[asset2]
#             _, pval, _  = coint(x,y)
#             if pval < 0.05:
#               cointegr.append([pval, asset1 + asset2])


# cointegr = pd.DataFrame(cointegr)
# cointegr.to_csv("cointegration.csv")

# print(cointegr)
# print("done")



# cur_date = indices[1]
# end_date = indices[-1]
# delta14 = timedelta(days=14)
# delta1 = timedelta(days=1)
# deltam1 = timedelta(days=-1)

# portfolio_trading_signal = pd.DataFrame(index = indices, columns = assets).fillna(0)
# initial = np.random.choice(assets, 5, replace = False) 
# df.loc[cur_date, initial] = 0.999

# start_date = cur_date
# while cur_date < start_date + timedelta(days=31):
#     cur_date += delta1
#     portfolio_trading_signal.loc[start_date, initial] = 1

# while cur_date <= end_date:
#     order = assets[df_lrt_30ma.loc[cur_date].argsort()]     
#     portfolio_trading_signal.loc[cur_date, order[:5]] = 0.999 #+ 0.01 ** portfolio_trading_signal.loc[(cur_date + deltam1), order[:5]]

#     start_date = cur_date
#     while cur_date < start_date + delta14 and cur_date < end_date:
#         cur_date += delta1
#         portfolio_trading_signal.loc[cur_date, order[:5]] = 1


# print(portfolio_trading_signal)




# def trade():



        

# mean_portfolio_trading_signal.index = stocks_df.index.unique()

# #fig = plt.figure(figsize=(8,12))
# fig = plt.figure(figsize=(8,12))
# fig.subplots_adjust(hspace=1.4, wspace=0.4)
# for i in range(1, 11, 1):
#     ax = fig.add_subplot(10, 2, i)
    
#     asset = ticker[i-1]
    
#     returns = pd.DataFrame(index = stocks_df.index.unique(), columns=['Buy and Hold', 'Strategy'])
#     tmp = stocks_df[stocks_df['ticker'] == asset]

#     returns['Buy and Hold'] = (tmp['returns'] - 1)
#     returns['Strategy'] = mean_portfolio_trading_signal[asset].fillna(0) * (tmp['returns'] - 1)

#     eqCurves = pd.DataFrame(index = stocks_df.index.unique(), columns=['Buy and Hold', 'Strategy'])
#     eqCurves['Buy and Hold'] = np.cumprod(returns['Buy and Hold'] + 1)
#     #portfolio_baseline[asset] = eqCurves['Buy and Hold']
#     eqCurves['Strategy'] = np.cumprod(returns['Strategy'] + 1)
#     #portfolio_strategy[asset] = eqCurves['Strategy']
    
#     ax.set_title = (asset)
#     lines = ax.plot(eqCurves['Buy and Hold'], color = 'blue')
#     lines = ax.plot(eqCurves['Strategy'], color = farbe[i-1])
#     #ax.set_title = ('Return Strategy vs. Buy and Hold ' + asset)
#     ax.title.set_text(asset)
