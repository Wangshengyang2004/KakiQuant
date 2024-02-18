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

    def fetch_kline_data(self, inst_id, bar):
        start_date = "2019-01-01"
        end_date = self.today()
        latest_timestamp_db = self.find_bounds(inst_id, bar)
        start_timestamp = latest_timestamp_db + 1 if latest_timestamp_db else int(pd.Timestamp(start_date).timestamp()) * 1000
        end_timestamp = int(pd.Timestamp(end_date).timestamp()) * 1000
        
        # Simplify for demonstration; implement your fetching logic here based on the API client

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
