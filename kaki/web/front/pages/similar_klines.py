import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from scipy.stats import spearmanr
import statsmodels.api as sm
from scipy import stats
import datetime 
from datetime import date
# 关闭通知
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger().setLevel(logging.ERROR)

import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from mpl_finance import candlestick_ochl
import mplfinance as mpf
from kaki.kkdatac.crypto import get_pairs, get_price
st.set_option('deprecation.showPyplotGlobalUse', False)

# Initialize session state
if "text_input" not in st.session_state:
    st.session_state.text_input = False
    st.session_state.disabled = False
if "option" not in st.session_state:
    st.session_state.option = "1D"
if "data" not in st.session_state:
    st.session_state.data = None
if "dt" not in st.session_state:
    st.session_state.dt = None
if "length" not in st.session_state:
    st.session_state.length = 60
if "d" not in st.session_state:
    st.session_state.d = datetime.date(2024, 1, 1)
if "topK" not in st.session_state:
    st.session_state.topK = 5
if "runed" not in st.session_state:
    st.session_state.runed = False
if "pairs" not in st.session_state:
    st.session_state.pairs = None

@st.cache_data
def prepare_data():
    # Crypto pairs list
    pairs = get_pairs("kline-1D")
    data = get_price(pairs, bar="1D",fields=['open','high','low','close','instId'])
    st.session_state.data = data
    st.session_state.data['date'] = st.session_state.data['timestamp'].dt.date
    st.session_state.data.drop(columns=['timestamp'],inplace=True)
    st.session_state.data.set_index(["instId","date"], inplace=True)
    return data

# @st.cache_resource
def load(bar="1D"):
    if st.session_state.data is not None and bar == '1D':
        return
    # Crypto pairs list
    pairs = get_pairs("kline-1D")
    st.toast(f"pairs loaded successfully. Totally: {len(pairs)}")
    st.session_state.data = get_price(pairs, bar="1D",fields=['open','high','low','close','instId'])
    st.success(f"Data loaded successfully. Totally: {len(st.session_state.data)}")
    

def get_previous_trading_date(end_date: str, n=1):
    end = date.fromisoformat(end_date)
    start = end - pd.Timedelta(days=n)
    return start, end

def k_line(data, instId, end_date, length=60):
    df = data.copy()
    start, end = get_previous_trading_date(end_date, length)
    print(f"start: {start}, end: {end}")
    stock_a = df.loc[instId].loc[start:end].copy()
    stock_a.index = date2num(stock_a.index)
    stock_a.index.names = ['date']
    quotes = stock_a[['open','close','high','low']].reset_index().set_index(['date','open','close','high','low']).index.tolist()

    if len(quotes) < 5:
        print("Not enough data to plot")
        return None
    fig,ax = plt.subplots(figsize = (10,6),facecolor = 'white')
    ax.xaxis_date()

    plt.yticks()
    plt.title(f'k-line-{instId}-{start}-{end}')
    plt.ylabel('price')

    fig = candlestick_ochl(ax,quotes,colorup = 'r',width = 0.6,colordown = 'g')
    plt.show()
    return True

def k_line_mpf(data:pd.DataFrame, instId:str, end_date:str, length=60):
    df = data.copy()
    # st.write(df)
    start, end = get_previous_trading_date(end_date, length)
    print(f"start: {start}, end: {end}")
    stock_a = df.loc[instId].loc[start:end].copy()
    # Datetime index
    stock_a.index = pd.to_datetime(stock_a.index)
    fig = mpf.plot(stock_a,type='candle',style='charles',mav=(5,10,20),volume=False,title=f'Crypto K-Line Chart:{instId}-{start}-{end})')
    st.pyplot(fig)

def cal_dt(data: pd.DataFrame, instid: str, end: str, length:int =60):
    with st.spinner(text="In progress..."):
        dt = pd.DataFrame(columns=['stock','startdate','enddate','T'])
        start_time,end_time = get_previous_trading_date(end,length)
        df = data
        stock_a = df.loc[instid].loc[start_time:end_time].copy()

        stocklist = sorted(set(data.index.get_level_values(0)))
        datelist = sorted(set(data.index.get_level_values(1)))

        y = 0
        # progress bar
        st.toast("Calculating correlation...")
        progress_text = "Calculation in progress. Please wait 3 mins."
        my_bar = st.progress(0, text=progress_text)
        lth = len(stocklist) * len(datelist[60:-20:20])
        for s in tqdm(stocklist):
            stock_b = data.loc[s]
            for d in datelist[60:-20:20]:
                end = d
                start = end_time - pd.Timedelta(days=60)
                temp_k_line = stock_b.loc[start:end].reset_index().iloc[:,1:]
                T = stock_a.reset_index().iloc[:,1:].corrwith(temp_k_line).mean()
                dt.loc[y] = [s,start,end,T]
                y += 1
                my_bar.progress(y/lth, text=progress_text)
        my_bar.empty()
        dt = dt.fillna(0)
        dt = dt.sort_values(by='T',ascending=False)
        st.success('Done!')
        return dt

def stats_dt(dt: pd.DataFrame):
    dt['T'].plot(kind = 'density', subplots = True, layout=(2,4), sharex = False , figsize = (20,10), fontsize = 15)
    st.pyplot()

def cal(data):
    # Date is ISO date
    dt = cal_dt(data, st.session_state.text_input, st.session_state.d)
    # If T > 0.998, drop the row
    dt = dt[dt['T'] < 0.998]
    st.session_state.dt = dt

def plot_topK(dt: pd.DataFrame, data: pd.DataFrame):  
    st.write("Top 5 similar klines")
    st.write(dt)
    i = 0
    for _ in range(st.session_state.topK):
        Tdt = dt.iloc[i]                         #获取相似度最高的值所在行
        stock = Tdt.stock                        #获取对应股票
        enddate = str(Tdt.enddate)                 #获取对应结束时间
        # print(stock,enddate)
        k_line_mpf(data,stock,enddate)

    st.session_state.runed = True

with st.sidebar:
    st.session_state.d = str(st.date_input("Date", datetime.date(2024, 1, 1)))
    st.session_state.text_input = st.text_input(
        "Your Crypto Pair (eg. XRP-USDT-SWAP) 👇",
        placeholder="XRP-USDT-SWAP",
        value="XRP-USDT-SWAP",
    )
    st.session_state.option = st.selectbox(
    "Bar",
    ("1D", "1m", "15m", "1H", "4H", "1W", "1M"),
    index=None,
    placeholder="Select kline bar",

    )
    st.session_state.length = st.number_input("Length", min_value=10, max_value=100, value=60)
    st.session_state.topK = st.number_input("Top K", min_value=1, max_value=10, value=5)

    if st.button("Load", type="primary"):
        load()
    if st.button("Reset", type="primary"):
        st.session_state.data = None
        st.session_state.dt = None
        st.session_state.d = None
        st.session_state.length = 60
        st.session_state.text_input = False
        st.session_state.disabled = False
        st.session_state.runed = False
        st.session_state.topK = 5


# Introduction
st.title("Similar K-Line")
st.write("""This page is used to find the top K similar klines of the given kline.The kline is based on the given date and the length of the kline.The top K similar klines are calculated based on the correlation between the given kline and other klines. The correlation is calculated based on the spearman correlation coefficient.The top K similar klines are plotted using the mplfinance library.""")
# Instructions
st.write("""Instructions:
1. Enter the date and the crypto pair.
2. Select the bar.
3. Enter the length of the kline.
4. Enter the top K value.
5. Click on the Load button to load the data.
6. Click on the Calculate button to calculate the top K similar klines.
7. The top K similar klines will be displayed below.
""")
st.session_state.data = prepare_data()
st.sidebar.write("Data loaded successfully. Totally: ", len(st.session_state.data))
if st.button("Calculate", type="primary"):
    cal(st.session_state.data)
if st.session_state.dt is not None:
    stats_dt(st.session_state.dt)
    if st.button("Plot", type="primary") and st.session_state.data is not None:
        plot_topK(st.session_state.dt, st.session_state.data)
        






