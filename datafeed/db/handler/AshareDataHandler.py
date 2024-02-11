import time
from BaseDataHandler import BaseDataHandler
import sys 
import os
import tushare as ts
import akshare as ak
# Add the parent directory of 'db' and 'utils' to the system path
script_dir = os.path.dirname(__file__)  # Path to the directory where the script is located
parent_dir = os.path.dirname(script_dir)  # Path to the parent directory
sys.path.append(parent_dir)
from utils.check_gpu import enable_cudf_acceleration
from utils.check_date import today
import logging 
# Try GPU acceleration
enable_cudf_acceleration()
import pandas as pd

class AshareDataHandler(BaseDataHandler):
    def __init__(self, db_name, collection_name):
        super().__init__(db_name, collection_name)
        self.pro = ts.pro_api(os.getenv["TUSHARE_TOKEN"])

    def fetch_index_components(self, index_code, start_date, end_date):
        """
        例子：获取指数成分股 沪深300
        :param index_code: str 指数代码
        :param start_date: str 开始日期
        :param end_date: str 结束日期
        :return: pandas.DataFrame
        数据库: MongoDB - Ashare_index_components - example: "000300.SH"
        
        
        """
        # Get the index components
        index_components = self.pro.index_member(index_code=index_code, start_date=start_date, end_date=end_date)
        return index_components
    
    def fetch_daily_klines(self, start_date=None, end_date=None, save_to_db:bool=True, return_df:bool=False):
        """
        更新每日行情数据，收盘后运行
        因研究需要，默认剔除ST股票，如果需要保留ST股票，请修改代码
        :param start_date: str 开始日期
        :param end_date: str 结束日期
        :return: pandas.DataFrame
        数据库: MongoDB - Ashare_daily_klines - example: "000300.SH"
        """
        end_date = today(tushare_format=True) if end_date is None else end_date
        start_date = "20100101" if start_date is None else start_date
        # 获取所有股票代码列表，以end_date为准
        stock_list = self.pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
        # Get the index components
        daily_klines = self.pro.daily(trade_date='', )
        return daily_klines
    
    def fetch_1min_klines(self, start_date=None, end_date=None, save_to_db:bool=True, return_df:bool=False):
        """
        更新每日1分钟行情数据，收盘后运行
        需要tushare pro 10000积分
        因研究需要，默认剔除ST股票，如果需要保留ST股票，请修改代码
        :param start_date: str 开始日期
        :param end_date: str 结束日期
        :return: pandas.DataFrame
        数据库: MongoDB - Ashare_daily_klines - example: "000300.SH"
        """
        end_date = today(tushare_format=True) if end_date is None else end_date
        start_date = "20100101" if start_date is None else start_date
        # 获取所有股票代码列表，以end_date为准
        stock_list = self.pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
        # Get the index components
        daily_klines = self.pro.daily(trade_date='', )
        return daily_klines