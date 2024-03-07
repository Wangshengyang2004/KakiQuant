import akshare as ak
from tqdm import tqdm
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
from kaki.utils.check_db import get_client_str

# MongoDB connection
client = MongoClient(get_client_str())  # Update with your MongoDB connection string
db = client["ashare_tushare"]  # Database name
collection = db["kline"]  # Collection name

def update_stock_data(stock_code, start_date, end_date):
    # Fetch new data for the specified range
    stock_data_df = ak.stock_zh_a_hist(symbol=stock_code, start_date=start_date, end_date=end_date, adjust="hfq")
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
        # Insert new data
        collection.insert_many(stock_data_df.to_dict('records'))
    
# Get stock list
stock_list_df = ak.stock_zh_a_spot_em()
stock_list = stock_list_df["代码"].tolist()

# Filtering out ST stocks
stock_zh_a_st_em_list = ak.stock_zh_a_st_em()["代码"].tolist()
stock_list_safe = [stock for stock in stock_list if stock not in stock_zh_a_st_em_list]

print(f"Total stocks: {len(stock_list_safe)}")

# List to store stocks with gold crossover
stocks_with_gold_crossover = []

for stock_code in tqdm(stock_list_safe):
    try:
        # Get or update stock daily data from MongoDB
        stock_data_df = get_stock_data_from_db(stock_code, "20200101", "20240118")
        if stock_data_df.empty:
            continue

        # Calculate MACD
        stock_data_df_with_macd = calculate_macd(stock_data_df)

        # Check for gold crossover
        if has_gold_crossover(stock_data_df_with_macd):
            stocks_with_gold_crossover.append(stock_code)
    except Exception as e:
        print(f"Error processing {stock_code}: {e}")

print(stocks_with_gold_crossover)
