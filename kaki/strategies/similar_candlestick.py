import pandas as pd
import numpy as np
import pickle
from tqdm import *
from scipy.stats import spearmanr
import statsmodels.api as sm
from scipy import stats

# 关闭通知
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger().setLevel(logging.ERROR)

import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from mplfinance.original_flavor import candlestick_ochl

from kaki.kkdatac.crypto import get_pairs, get_price
from datetime import date



def k_line(data, instId, end_date, length=60):
    df = data.copy()
    end = date.fromisoformat(end_date)
    start = end - pd.Timedelta(days=60)
    logging.info(f"start: {start}, end: {end}")
    stock_a = df.loc['BTC-USDT-SWAP'].loc[start:end].copy()
    stock_a.index = date2num(stock_a.index)
    stock_a.index.names = ['date']
    quotes = stock_a[['open','close','high','low']].reset_index().set_index(['date','open','close','high','low']).index.tolist()

    fig,ax = plt.subplots(figsize = (10,6),facecolor = 'white')
    ax.xaxis_date()

    plt.yticks()
    plt.title('k-line')
    plt.ylabel('price')

    candlestick_ochl(ax,quotes,colorup = 'r',width = 0.6,colordown = 'g')
    plt.show()


if __name__ == '__main__':
    # Crypto pairs list
    pairs = get_pairs("kline-1D")

    data = get_price(pairs, bar="1D",fields=['open','high','low','close','instId'])
    print(data)
    data['date'] = data['timestamp'].dt.date
    data.drop(columns=['timestamp'],inplace=True)
    data = data.set_index(["instId","date"])
    k_line(data, 'BTC-USDT-SWAP', '2022-09-01', length=60)
    