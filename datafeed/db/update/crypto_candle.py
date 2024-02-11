import okx.MarketData as MarketData
import pandas as pd
import time
from .mongo_middleware import MongoDBHandler
import random
import logging
from ..handler.util import timestamp_to_readable_date
logging.basicConfig(level=logging.INFO)


def connect_to_mongodb(db_name, inst_id, bar):
    collection_name = f"{inst_id}-{bar}".replace('/', '-').replace(' ', '')  # Format collection name
    client = MongoDBHandler(db_name, collection_name)  # Connect to the MongoDB server

    return client.collection

def insert_data_to_mongodb(collection, data):
    collection.insert_many(data.to_dict('records'))

def process_data(data):
    # This function will process your data as needed
    # Add your data processing steps here
        # Process and insert data to MongoDB
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote', 'confirm'])
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    df['Date'] = df['timestamp'].dt.strftime('%Y%m%d%H%M%S')
    # Convert other fields to appropriate numeric types
    numeric_fields = ['open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote', 'confirm']
    for field in numeric_fields:
        df[field] = pd.to_numeric(df[field], errors='coerce')
    return df

def fetch_crypto_klines(start_date, end_date, inst_id, bar, max_retries=3, initial_delay=1, save_to_db=True):
    flag = "0"  # Production trading:0 , demo trading:1
    marketDataAPI = MarketData.MarketAPI(flag=flag)

    # MongoDB Connection
    collection = connect_to_mongodb("crypto_candle", inst_id, bar)

     # Convert start and end dates to Unix timestamp in milliseconds
    start_timestamp = int(pd.Timestamp(start_date).timestamp()) * 1000
    end_timestamp = int(pd.Timestamp(end_date).timestamp()) * 1000

    # Initially, 'before' is None to fetch the latest data
    a = end_timestamp
    b = None
    is_First_time = True
    while True:
        retries = 0
        while retries < max_retries:
            try:
                result = marketDataAPI.get_history_candlesticks(
                    instId=inst_id,
                    before=str(b) if b else "",
                    after=str(a),
                    bar=bar
                )

                # Check if result is empty or contains data
                if not result['data']:
                    print("No more data to fetch or empty data returned.")
                    return

                # Process and insert data to MongoDB
                df = process_data(result['data'])
                insert_data_to_mongodb(collection, df)

                # Update 'before' and 'after' for the next request
                earliest_timestamp = int(result['data'][-1][0])
                latest_timestamp = int(result['data'][0][0])
                if is_First_time:
                    time_interval = latest_timestamp - earliest_timestamp
                    is_First_time = False
                a = earliest_timestamp
                b = a - time_interval - 4 + random.randint(1, 10)*2

                if a <= start_timestamp:
                    print("Reached the start date.")
                    return

                break

            except Exception as e:
                print(f"Error occurred: {e}")
                retries += 1
                time.sleep(initial_delay * retries)  # Exponential backoff
                if retries == max_retries:
                    print("Max retries reached. Exiting.")
                    return
                print(f"Retrying... Attempt {retries}/{max_retries}")


if __name__ == "__main__":
    # Example usage
    fetch_crypto_klines(start_date="2020-01-01", end_date="2024-01-01", inst_id="SATS-USDT-SWAP", bar="1D",max_retries=3, initial_delay=1)
