import pandas as pd
import pymongo
from datetime import datetime
import logging
import time
import concurrent.futures
import okx.MarketData as MarketData
import okx.PublicData as PublicData
from pymongo import MongoClient

class CryptoDataUpdater:
    def __init__(self):
        self.client = MongoClient('192.168.31.120', 27017)
        self.db = self.client.crypto
        self.collection = self.db.crypto_kline
        self.api_client = MarketData.MarketAPI(flag="0")  # Adjust this based on your API client initialization
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", 
                            handlers=[logging.FileHandler("crypto_AIO.log"), logging.StreamHandler()])

    def find_bounds(self, inst_id, bar):
        latest_record = self.collection.find_one({'instId': inst_id, 'bar': bar}, sort=[('timestamp', pymongo.DESCENDING)])
        if latest_record:
            return latest_record['timestamp']
        return None

    def insert_data_to_mongodb(self, data):
        if not data.empty:
            data_dict = data.to_dict('records')
            self.collection.insert_many(data_dict)
            logging.info(f"Inserted {len(data_dict)} new records.")

    def get_all_coin_pairs(self):
        publicDataAPI = PublicData.PublicAPI(flag="0")
        spot_result = publicDataAPI.get_instruments(instType="SPOT")
        swap_result = publicDataAPI.get_instruments(instType="SWAP")
        spot_list = [i["instId"] for i in spot_result["data"]]
        swap_list = [i["instId"] for i in swap_result["data"]]
        return spot_list + swap_list

    def process_kline(self, data):
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote'])
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        numeric_fields = ['open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote']
        for field in numeric_fields:
            df[field] = pd.to_numeric(df[field], errors='coerce')
        return df

    def fetch_kline_data(self, inst_id, bar, start_date, end_date, max_retries=3, initial_delay=1, save_to_db=True):
        # Convert start and end dates to Unix timestamp in milliseconds
        start_timestamp = int(pd.Timestamp(start_date).timestamp()) * 1000
        end_timestamp = int(pd.Timestamp(end_date).timestamp()) * 1000

        # Initially, 'before' is None to fetch the latest data
        a = end_timestamp
        b = None
        is_first_time = True
        while True:
            retries = 0
            while retries < max_retries:
                try:
                    result = self.api_client.get_history_candlesticks(
                        instId=inst_id,
                        before=str(b) if b else "",
                        after=str(a),
                        bar=bar
                    )

                    # Check if result is empty or contains data
                    if not result['data']:
                        logging.info("No more data to fetch or empty data returned.")
                        return

                    # Process the data
                    df = self.process_kline(result['data'])
                    
                    # Insert data to MongoDB if applicable
                    if save_to_db and not df.empty:
                        self.insert_data_to_mongodb(df)
                        logging.info(f"Successfully inserted data for {inst_id} {bar}.")

                    # Update 'before' and 'after' for the next request
                    earliest_timestamp = int(result['data'][-1][0])
                    if is_first_time:
                        time_interval = int(result['data'][0][0]) - earliest_timestamp
                        is_first_time = False
                    a = earliest_timestamp - time_interval - 1

                    # Break if we have reached the starting timestamp
                    if a <= start_timestamp:
                        logging.info(f"Reached the start date for {inst_id} {bar}.")
                        return

                    break

                except Exception as e:
                    logging.error(f"Error occurred: {e}")
                    retries += 1
                    time.sleep(initial_delay * retries)  # Exponential backoff
                    if retries == max_retries:
                        logging.error("Max retries reached. Exiting.")
                        return
                    logging.info(f"Retrying... Attempt {retries}/{max_retries}")


    @staticmethod
    def today(tushare_format=False):
        if tushare_format:
            return datetime.today().strftime("%Y%m%d")
        return datetime.today().strftime("%Y-%m-%d")

    def update_data(self):
        pair_list = self.get_all_coin_pairs()
        bar_sizes = ['1m', '3m', '5m', '15m', '30m', '1H', '4H', '1D', '1W']
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for inst_id in pair_list:
                for bar in bar_sizes:
                    futures.append(executor.submit(self.fetch_kline_data, inst_id, bar))
            concurrent.futures.wait(futures)

if __name__ == "__main__":
    updater = CryptoDataUpdater()
    updater.update_data()
