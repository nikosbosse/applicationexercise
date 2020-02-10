import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil import parser
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
        ## function lets time between two trades pass and increments the 
        ## current date. In theory, this function could also trigger 
        ## stop loss orders 
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
        ## this function decides which stocks to hold according
        ## to a trading strategy. It then executes the trade
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
        ## simulates selling everything
        # cur value = cash + what I get if I sell everything
        new_cur_value = self.cash + np.sum(self.df.loc[self.current_date, self.endowment]) * (1 - self.tradecosts)

        new_cur_value_baseline = self.cash_baseline + np.sum(self.df.loc[self.current_date, :]) * (1 - self.tradecosts)

        self.cur_value = new_cur_value
        self.cur_value_baseline = new_cur_value_baseline
        self.results.append([self.current_date, self.cur_value, self.cur_value_baseline])

    ## functions for plotting and analysis. Allows to access the results
    ## directly as methods. 

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

    ## wrapper around the analysis functions
    def run_analysis(self):
        f.plot_all(self.df, self.file)
        f.plot_all_log(self.df, self.file)
        f.plot_autocorrelations(self.df_lrt, self.file)
        f.examine_stationarity(self.df_lrt, self.file)
        f.examine_cointegration(self.df, self.file)
        f.plot_autocorrelations(self.df_lrt ** 2, file = self.file + "squared")
        f.select_arma_order(self.df_lrt, self.file)

    ## directly gets out the results for the trading strategy. 
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

#df_train = df.iloc[0:1500, :]
# df_train = df_train.drop(['Asset2', 'Asset25'], axis = 1)

backtest = backtest(df_train, file_id = "train_")
#backtest = backtest(df_train, file_id = "train_", strategy = "mean_reversion")
#backtest = backtest(df_train, file_id = "train_", strategy = "mean_reversion")

while (not backtest.finish):
    backtest.trade()
    backtest.increment_time()
    backtest.track_profits()

backtest.run_analysis()
backtest.examine_strategies()
print(backtest.results)
plt.show()
