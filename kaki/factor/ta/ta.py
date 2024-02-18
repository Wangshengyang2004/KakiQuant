import numpy as np 
import pandas as pd
import talib
import os
import warnings
warnings.filterwarnings("ignore")

# Function to calculate Simple Moving Average
def SMA(data, period=30, column='close'):
    return data[column].rolling(window=period).mean()

# Function to calculate Exponential Moving Average
def EMA(data, period=20, column='close'):
    return data[column].ewm(span=period, adjust=False).mean()

# Function to calculate the Relative Strength Index (RSI)
def RSI(data, period=14, column='close'):
    delta = data[column].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    RS = gain / loss
    return 100 - (100 / (1 + RS))
    
# Function to calculate the Bollinger Bands
def BBANDS(data, period=20, column='close'):
    sma = SMA(data, period, column=column)
    std_dev = data[column].rolling(window=period).std()
    bollinger_up = sma + (std_dev * 2) # Upper Band
    bollinger_down = sma - (std_dev * 2) # Lower Band
    return bollinger_up, bollinger_down

# Function to calculate the Average True Range (ATR)
def ATR(data, period=14):
    data['high-low'] = data['high'] - data['low']
    data['high-close'] = np.abs(data['high'] - data['close'].shift(1))
    data['low-close'] = np.abs(data['low'] - data['close'].shift(1))
    ranges = data[['high-low', 'high-close', 'low-close']].max(axis=1)
    return ranges.rolling(period).mean()


#计算技术指标
def cal_ta_factors(df):
    df["DEMA"] = talib.DEMA(df["close"], timeperiod=30)
    df["SMA"] = talib.SMA(df["close"], timeperiod=30)
    df["EMA30"] = talib.EMA(df["close"], timeperiod=30)
    df["EMA40"] = talib.EMA(df["close"], timeperiod=40)
    df["EMA50"] = talib.EMA(df["close"], timeperiod=50)
    df["EMA60"] = talib.EMA(df["close"], timeperiod=60)
    df["EMA70"] = talib.EMA(df["close"], timeperiod=70)
    df["KAMA"] = talib.KAMA(df["close"], timeperiod=30)
    df["SAR"] = talib.SAR(df["high"], df["low"], acceleration=0.02, maximum=0.2)
    df["T3"] = talib.T3(df["close"], timeperiod=5, vfactor=0)
    df["TEMA"] = talib.TEMA(df["close"], timeperiod=30)
    df["TRIMA"] = talib.TRIMA(df["close"], timeperiod=30)
    df["MIDPRICE"] = talib.MIDPRICE(df["high"], df["low"], timeperiod=14)
    df["SAREXT"] = talib.SAREXT(df["high"], df["low"], startvalue=0, offsetonreverse=0, accelerationinitlong=0.02, accelerationlong=0.02, accelerationmaxlong=0.2, accelerationinitshort=0.02, accelerationshort=0.02, accelerationmaxshort=0.2)
    df["NATR"] = talib.NATR(df["high"], df["low"], df["close"], timeperiod=14)
    df["TRANGE"] = talib.TRANGE(df["high"], df["low"], df["close"])
    df["AD"] = talib.AD(df["high"], df["low"], df["close"], df["volume"])
    df["ADOSC"] = talib.ADOSC(df["high"], df["low"], df["close"], df["volume"], fastperiod=3, slowperiod=10)
    df["OBV"] = talib.OBV(df["close"], df["volume"])
    df["HT_DCPERIOD"] = talib.HT_DCPERIOD(df["close"])
    df["HT_PHASOR_INPHASE"], df["HT_PHASOR_QUADRATURE"] = talib.HT_PHASOR(df["close"])
    df["VAR"] = talib.VAR(df["close"], timeperiod=5, nbdev=1)
    df["TSF"] = talib.TSF(df["close"], timeperiod=14)
    df["BETA"] = talib.BETA(df["high"], df["low"], timeperiod=5)
    df["WILLR"] = talib.WILLR(df["high"], df["low"], df["close"], timeperiod=14)
    df["ULTOSC"] = talib.ULTOSC(df["high"], df["low"], df["close"], timeperiod1=7, timeperiod2=14, timeperiod3=28)
    df["TRIX"] = talib.TRIX(df["close"], timeperiod=30)
    df["RSI"] = talib.RSI(df["close"], timeperiod=14)
    df["MOM"] = talib.MOM(df["close"], timeperiod=10)
    df["ROC"] = talib.ROC(df["close"], timeperiod=10)
    df["PPO"] = talib.PPO(df["close"], fastperiod=12, slowperiod=26, matype=0)
    df["MFI"] = talib.MFI(df["high"], df["low"], df["close"], df["volume"], timeperiod=14)
    df["MACD_MACD"], df["MACD_MACDSIGNAL"], df["MACD_MACDHIST"] = talib.MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)
    df["DX"] = talib.DX(df["high"], df["low"], df["close"], timeperiod=14)
    df["CMO"] = talib.CMO(df["close"], timeperiod=14)
    df["CCI"] = talib.CCI(df["high"], df["low"], df["close"], timeperiod=14)
    df["ADX"] = talib.ADX(df["high"], df["low"], df["close"], timeperiod=14)
    df['Bollinger_Up'], df['Bollinger_Down'] = BBANDS(df, period=20)
    df['ATR_14'] = ATR(df, period=14)

    return df