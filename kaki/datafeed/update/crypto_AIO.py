import pandas as pd
import pymongo
import logging
import time
import concurrent.futures
import requests
import okx.PublicData as PublicData
from pymongo import MongoClient
import random
import numpy as np
from typing import Optional, Union

class CryptoDataUpdater:
    def __init__(self):
        self.client = MongoClient('localhost', 27017)
        self.db = self.client.crypto
        self.collection = self.db.crypto_kline
        self.base_url = "https://www.okx.com/api/v5/market/history-candles"
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", 
                            handlers=[logging.FileHandler("crypto_AIO.log"), logging.StreamHandler()])
        
        self.session = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive'
        }

    def set_mongodb_index(self) -> None:
        self.collection.create_index([("instId", pymongo.ASCENDING), ("bar", pymongo.ASCENDING), ("timestamp", pymongo.ASCENDING)], unique=True)

    def find_bounds(self, inst_id:str, bar:str) -> tuple:
        latest_record = self.collection.find_one({'instId': inst_id, 'bar': bar}, sort=[('timestamp', pymongo.DESCENDING)])
        earlist_record = self.collection.find_one({'instId': inst_id, 'bar': bar}, sort=[('timestamp', pymongo.ASCENDING)])
        if latest_record and earlist_record:
            # a, b
            return latest_record['timestamp'], earlist_record['timestamp']
        return None, None
    
    def preprocess(self):
        self.set_mongodb_index()
        pass

    def postprocess(self):
        self.data_clean()
        pass

    def data_clean(self):
        pass

    def insert_data_to_mongodb(self, data: pd.DataFrame):
        if not data.empty:
            data_dict = data.to_dict('records')
            self.collection.insert_many(data_dict)
            logging.info(f"Inserted {len(data_dict)} new records.")

    def get_all_coin_pairs(self):
        publicDataAPI = PublicData.PublicAPI(flag="0")
        spot_result = publicDataAPI.get_instruments(instType="SPOT")
        swap_result = publicDataAPI.get_instruments(instType="SWAP")
        return [i["instId"] for i in spot_result["data"]] + [i["instId"] for i in swap_result["data"]]
    
    def process_kline(self, data: dict, inst_id: str, bar: str) -> pd.DataFrame:
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote', 'confirm'])
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(np.int64), unit='ms')
        numeric_fields = ['open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote', 'confirm']
        for field in numeric_fields:
            df[field] = pd.to_numeric(df[field], errors='coerce')
        df['instId'] = inst_id
        df['bar'] = bar
        return df

    def newest_data_ts(self, inst_id: str, bar: str) -> np.int64:
        params = {
            'instId': inst_id,
            'bar': bar,
            'limit': 1
        }
        result = self.make_request(params=params).json(parse_int=np.int64)["data"][0][0]
        logging.info(f"Newest data for {inst_id}-{bar} is at {self.ts_to_date(result)}.")
        return result

    def ts_to_date(self, ts: np.int64) -> str:
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(float(ts) / 1000))
    
    def make_request(self, params: dict) -> requests.Response:
        response = requests.get(self.base_url, params=params, headers=self.headers, timeout=3)
        return response
    
    def convert_ints_to_np_int64(self, obj):
        if isinstance(obj, dict):
            return {k: self.convert_ints_to_np_int64(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_ints_to_np_int64(v) for v in obj]
        elif isinstance(obj, int):
            return np.int64(obj)
        else:
            return obj
    
    def fetch_candlesticks(self, inst_id: str, bar: str, before_ts: np.int64, after_ts: np.int64, sleep_time: int = 1, limit: int = 100) -> Optional[pd.DataFrame]:
        latest_ts = after_ts if after_ts else self.newest_data_ts(inst_id, bar)
        earliest_ts = before_ts
        a = latest_ts
        b = np.int64(a-1)
        
        is_first_time = True
        while True and b > earliest_ts:
            try:
                params = {
                    'instId': inst_id,
                    'before': "" if is_first_time else str(b),
                    'after': str(a),
                    'bar': bar,
                    'limit': limit
                }
                response = self.make_request(params)
                if response.status_code == 200:
                    result = response.json(parse_int=np.int64)
                    if not result['data']:
                        logging.info(f"No more data to fetch or empty data returned for {inst_id}-{bar}.")
                        return None
                    else:
                        df = self.process_kline(result['data'], inst_id=inst_id, bar=bar)
                        if not df.empty:
                            self.insert_data_to_mongodb(df)
                            logging.info(f"Successfully inserted data for {inst_id} {bar}.")
                        a = int(result['data'][-1][0])
                        if is_first_time:
                            time_interval = int(result['data'][0][0]) - a
                            is_first_time = False
                        b = a - time_interval - np.int64(4) + random.randint(1, 10)*2
                
                elif response.status_code == 429:
                    logging.info(f"Too many requests. Sleeping for {sleep_time:.2f} seconds.")
                    time.sleep(sleep_time)
                
                else:
                    logging.error(f"Failed to fetch data: {result.text}")
                    return None
            
            except Exception as e:
                logging.error(f"Error occurred: {e}")
                time.sleep(sleep_time)
    
    def check_missing_data_to_fill(self, inst_id: str, bar: str) -> Optional[Union[dict, None]]:
        """
        Check if there is any missing data for the given inst_id and bar.
        If there is, return the missing data range in the form of a dict.
        """
        pass

    def fetch_kline_data(self, inst_id: str, bar: str, initial_delay: int = 1) -> None:
        # Check the missing gap first:
        missing_dict = self.check_missing_data_to_fill(inst_id, bar)
        #### NEED TO IMPLEMENT THIS FUNCTION

        # submit the job to concurrent
 

    def update_data(self) -> None:
        pair_list = self.get_all_coin_pairs()
        # pair_list = ["BTC-USDT-SWAP"]
        bar_sizes = ['1m', '3m', '5m', '15m', '30m', '1H', '4H', '1D', '1W']
        # bar_sizes = ['1m']
        # self.fetch_kline_data(inst_id=pair_list[0], bar=bar_sizes[0])
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for inst_id in pair_list:
                for bar in bar_sizes:
                    futures.append(executor.submit(self.fetch_kline_data, inst_id, bar))
            concurrent.futures.wait(futures)

if __name__ == "__main__":
    updater = CryptoDataUpdater()
    updater.update_data()
