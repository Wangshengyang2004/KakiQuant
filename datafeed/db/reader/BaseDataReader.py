"""
返回DataLoader, 可以获取数据库的时间序列，得到DataFrame
"""
import pandas as pd
import os
from pymongo import MongoClient
import logging
from dotenv import load_dotenv
from ..handler.util import get_client_str

load_dotenv("../config/config.env")

class BaseDataReader:
    def __init__(self):
        pass

    def get_data(self, start_date, end_date):
        raise NotImplementedError

    def get_df(self, start_date, end_date):
        raise NotImplementedError

    def get_data_loader(self, start_date, end_date):
        raise NotImplementedError
    
    def use_sql(self, sql):
        raise NotImplementedError
    
    def min_date(self):
        raise NotImplementedError
    
    def max_date(self):
        raise NotImplementedError