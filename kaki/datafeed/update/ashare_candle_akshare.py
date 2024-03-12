import akshare as ak
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
from kaki.utils.check_db import get_client_str
from concurrent.futures import ThreadPoolExecutor



def drop_db(client):
    client.drop_database('ashare_tushare')

def insert_pd_to_db(collection, data):
    collection.insert_many(data.to_dict('records'))

def download_one_full(stock_code, freq:str ='1D'):
    collection = db[f'kline-{freq}']
    # Fetch data for the specified range or full data if no start date or end date is provided
    if freq == "1D":
        stock_data_df = ak.stock_zh_a_hist(symbol=stock_code, adjust="hfq")
    elif freq == '1W':
        stock_data_df = ak.stock_zh_a_hist(symbol=stock_code, period='weekly',adjust="hfq")
    else:
        stock_data_df = ak.stock_zh_a_minute(symbol=stock_code, period=freq[0:-1], adjust="qfq")
   
    if not stock_data_df.empty:
        # Convert '日期' column to datetime.datetime
        stock_data_df['日期'] = pd.to_datetime(stock_data_df['日期'])

        # Convert '日期' column to a format suitable for MongoDB
        stock_data_df['date'] = stock_data_df['日期'].apply(lambda x: datetime.combine(x, datetime.min.time()))

        # Check if conversion is successful
        if not all(isinstance(d, datetime) for d in stock_data_df['date']):
            print(f"Data type issue in stock {stock_code}")
            return

        stock_data_df['symbol'] = stock_code
        # Insert data
        insert_pd_to_db(collection, stock_data_df)

def download_stocks(stock_list, num_workers=5):
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        bar_sizes = ['1D', '1W']
        for stock_code in stock_list:
            for bar in bar_sizes:
                futures.append(executor.submit(download_one_full, stock_code, freq=bar))
        for future in futures:
            future.result()

if __name__ == "__main__":
    # MongoDB connection
    client = MongoClient(get_client_str())  # Update with your MongoDB connection string
    print(get_client_str())
    drop_db(client=client)
    db = client["ashare_tushare"]  # Database name
    # Get stock list
    stock_list_df = ak.stock_zh_a_spot_em()
    stock_list = stock_list_df["代码"].tolist()

    # Download full data for stocks using workers
    download_stocks(stock_list)
