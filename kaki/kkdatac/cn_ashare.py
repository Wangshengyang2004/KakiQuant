"""
High Level Wrapper for kaki datafeed, providing similar API as ricequant for CN_Ashare 
"""
import pandas as pd
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv
from typing import List
load_dotenv("../config/db.env")

# 获取指数成分
def index_components(index_code: str) -> List:
    pass

# 获取个股信息
def get_price(stock_code, start_date, end_date, frequency, fields) -> pd.DataFrame:
    pass

def get_trading_dates(start_date=None, end_date=None) -> pd.TimedeltaIndex:
    pass

def get_previous_trading_date(this_date:str, nums_prev:int):
    pass

def get_ashare_price():
    pass