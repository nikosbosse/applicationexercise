# =================================== #
#               utilities             #
# =================================== #

def load_data():
    import pandas as pd
    from datetime import datetime

    df = pd.read_csv("../data/data.csv")
    df['Date'] = pd.to_datetime(df['Unnamed: 0'])
    df.index = df['Date']
    df = df.drop(['Date', 'Unnamed: 0'], axis = 1)
    df = df.reindex(index=df.index[::-1])
    return(df)

def make_log_returns(df):
    import pandas as pd
    import numpy as np

    df_lrt = df.copy()
    for asset in df_lrt.columns:
        df_lrt.loc[:, asset] = np.log(df[asset] / df[asset].shift(1))
    return(df_lrt.iloc[1:,:])


def make_returns(df):
    import pandas as pd
    import numpy as np

    df_rt = df.copy()
    for asset in df_rt.columns:
        df_rt.loc[:, asset] = (df[asset] / df[asset].shift(1))
    return(df_rt.iloc[1:,:])


def get_moving_average(df_lrt, N = 30):
    df_lrt_ma = df_lrt.copy()
    for asset in df_lrt.columns:
        df_lrt_ma.loc[:, asset] = df_lrt[asset].rolling(window=N).mean().values
    return(df_lrt_ma)


# =================================== #
#            Diagnostics              #
# =================================== #

def examine_stationarity(df_lrt, file):
    from statsmodels.tsa.stattools import adfuller
    import pandas as pd
    res = []
    for asset in df_lrt.columns:
        tmp1 = (adfuller(df_lrt[asset], regression="c")[1])
        tmp2 = (adfuller(df_lrt[asset], regression="nc")[1])
        tmp3 = (adfuller(df_lrt[asset], regression="ct")[1])
        tmp4 = (adfuller(df_lrt[asset], regression="ctt")[1])
        res.append([asset, tmp1, tmp2, tmp3, tmp4])
    res = pd.DataFrame(res)
    res.columns = ['Asset', 'adfuller_const', 'adfuller_noconst', 'adfuller_const_trend', 'adfuller_all']
    res.to_csv("../results/" + file + "stationarity.csv") 
    return(res)
    print("look at stationarity done")


def examine_cointegration(df, file):
    from statsmodels.tsa.stattools import coint
    import pandas as pd
    cointegr = []
    for asset1 in df.columns:
        for asset2 in df.columns:
            if asset1 != asset2:
                x = df[asset1]
                y = df[asset2]
                _, pval, _  = coint(x,y)
                if pval < 0.05:
                  cointegr.append([pval, asset1 + asset2])
    cointegr = pd.DataFrame(cointegr)
    cointegr.to_csv("../results/" + file + "cointegration.csv")
    return(cointegr)
    print("look for cointegration done")


def select_arma_order(df_lrt, file):
    from statsmodels.tsa.stattools import arma_order_select_ic
    import pandas as pd
    res = []
    for asset in df_lrt.columns:
        try:
            tmp = arma_order_select_ic(df_lrt[asset])
            res.append(tmp)
        except:
            pass

    res = pd.DataFrame(res)
    res.to_csv("../results/" + file + "select_arma_order.csv")
    return(res)
    print("find ARMA order done")


# =================================== #
#               Plotting              #
# =================================== #

def plot_all_log(df, file):
    import matplotlib.pyplot as plt
    import numpy as np
    fig = plt.figure(figsize=(10, 10))
    for asset in df.columns:
        plt.plot(np.log(df[asset])) 
    fig.savefig("../results/" + file + "all_assets_logged.jpg")
    print("plot all log assets done")


def plot_all(df, file):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 10))
    for asset in df.columns:
        plt.plot((df[asset])) 
    fig.savefig("../results/" + file + "all_assets.jpg")
    print("plot all assets done")


def plot_autocorrelations(df_lrt, file):
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16,80))
    fig.subplots_adjust(hspace=1.4, wspace=0.4)
    for i in range(1, (2 * 27 + 1), 2):
        ax1 = fig.add_subplot(27, 2, i)
        ax2 = fig.add_subplot(27, 2, i+1)
        ind = int(i/2 - 0.5)
        try:
            sm.graphics.tsa.plot_acf(df_lrt[df_lrt.columns.values[ind]], lags=15, ax = ax1)
            sm.graphics.tsa.plot_pacf(df_lrt[df_lrt.columns.values[ind]], lags=15, ax = ax2)
            ax1.set_title("Autocorrelation: Asset " + str(ind))
            ax2.set_title("Partial Autocorrelation Asset " + str(ind))
        except:
            pass
   
    fig.savefig("../results/" + file + "all_autocorr_log_returns.jpg", bbox_inches='tight')
    print("autocorrelations done")



# =================================== #
#           Predict / trade           #
# =================================== #


def get_best_performers(df_rt_ma, time, n = 8):
    tmp = df_rt_ma.loc[time].argsort()
    return(df_rt_ma.columns[tmp][-n:])


def get_worst_performers(df_rt_ma, time, n = 8):
    tmp = df_rt_ma.loc[time].argsort()
    return(df_rt_ma.columns[tmp][:(n-1)])


def get_best_predicted(df_lrt, current_date):
    # Pseudocode
    # get predictions
    # pick n stocks with the best predictions
    # or trade on those that have a prediction
    # value larger than a threshold
    return ()


def predict_ARMA(df_lrt, time_index, days_to_forecast):
    import pandas as pd
    import statsmodels.tsa.api as smt
    import numpy as np
    import sys, os

    # block printing  did not work. 
    # sys.stdout = open(os.devnull, 'w')

    df = df_lrt.iloc[(time_index - 30):time_index, :]
    prediction = np.empty(len(df.columns))

    i = 0
    ind = time_index
    for asset in df.columns:
        tmp = df[asset]
        try: 
            model = smt.ARMA(df[asset], (1,1))
            model_fit = model.fit() 
        except:
            model = smt.ARMA(df[asset], (0,0))
            model_fit = model.fit()
        prediction[i] = model_fit.forecast(steps = days_to_forecast)[0][-1]
        i+=1
    ## could be improved by implementing auto arma to find ar and ma values. 

    res = prediction
    res = (prediction - df.iloc[-1, :].values) / df.std(axis = 0) 

    #reenable printing
    # sys.stdout = sys.__stdout__

    return (res)











