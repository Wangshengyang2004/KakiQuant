"""
Update the crypto data in MongoDB asynchronously using aiohttp and asyncio.
Full data, and will run everyday to update the data.
"""
import pandas as pd
from datetime import datetime
import logging
import asyncio
import random
import aiohttp
import okx.PublicData as PublicData
from motor.motor_asyncio import AsyncIOMotorClient
import numpy as np
from typing import Optional, Iterable
import aiohttp
from collections.abc import Sequence
from kaki.utils.check_db import insert_data_to_mongodb
from kaki.utils.check_root_base import find_and_add_project_root
from omegaconf import OmegaConf
from pymongo.errors import BulkWriteError
import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", 
                    handlers=[logging.FileHandler("crypto_fetch.log"), logging.StreamHandler()])

# Create a TypeAlias for timestamp format
TIMESTAMP = np.int64

conf = OmegaConf.load(f"{find_and_add_project_root()}/config/config.yaml")
db_str = "mongodb://" + conf.db.mongodb.host + ":" + str(conf.db.mongodb.port)
logging.debug(conf)
bar_sizes = conf.market.crypto.bar.interval

class AsyncCryptoDataUpdater:
    def __init__(self, bar_sizes: Iterable[str] = bar_sizes, 
                 max_concurrent_requests:int = 5) -> None:
        self.client = AsyncIOMotorClient(db_str)
        self.db = self.client.crypto
        self.bar_sizes = bar_sizes
        self.market_url = "https://www.okx.com/api/v5/market/history-candles"
        self.headers = {
            "User-Agent": "PostmanRuntime/7.36.3",
            "Accept": "*/*",
            "b-locale": "zh_CN",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
            "Connection": "keep-alive",
            "Host": "www.okx.com",
            "Referer": "https://www.okx.com/",
        }
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", 
                            handlers=[logging.FileHandler("aiocrypto.log"), logging.StreamHandler()])
    
    async def drop_db(self):
        """
        Function to drop the whole db, here db: crypto
        """
        await self.client.drop_database('crypto')
        logging.info("Dropped database 'crypto'.")

    async def start_session(self):
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=10)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def close_session(self):
        if self.session and not self.session.closed:
            await self.session.close()

    async def create_collections_and_indexes(self):
        desired_col_list = [f"kline-{i}" for i in self.bar_sizes]
        # Get the list of existing collections in the database
        existing_collections = await self.db.list_collection_names()
        
        for collection_name in desired_col_list:
            # Check if the collection already exists
            if collection_name not in existing_collections:
                # If the collection does not exist, create it by inserting a dummy document
                await self.db[collection_name].insert_one({'init': True})
                logging.info(f"Collection {collection_name} created with a dummy document.")
            else:
                logging.info(f"Collection {collection_name} already exists.")
            
            # Check and create indexes
            await self.ensure_indexes(collection_name)

    async def ensure_indexes(self, collection_name):
        collection = self.db[collection_name]
        # Fetch the current indexes on the collection
        current_indexes = await collection.index_information()
        
        # Define your desired index key pattern
        desired_index_key = [("instId", 1), ("timestamp", 1)]
        
        # Determine if your desired index already exists
        index_exists = any(index['key'] == desired_index_key for index in current_indexes.values())
        
        if not index_exists:
            # Create the index if it does not exist
            await collection.create_index(desired_index_key, unique=True)
            logging.info(f"Index on {desired_index_key} created for collection {collection_name}.")
        else:
            logging.info(f"Index on {desired_index_key} already exists for collection {collection_name}.")

    


    async def get_all_coin_pairs(self, filter: Optional[str] = None) -> list[str]:
        """
        Get all coin pairs from the OKEx API.
        Filter out the coin pairs with Regex.
        """
        # Wrap the synchronous calls in asyncio.to_thread to run them in separate threads
        spot_result = await asyncio.to_thread(self._get_spot_instruments)
        swap_result = await asyncio.to_thread(self._get_swap_instruments)
        
        spot_list = [i["instId"] for i in spot_result["data"]]
        swap_list = [i["instId"] for i in swap_result["data"]]
        all_coin_pairs = spot_list + swap_list
        if filter:
            return [pair for pair in all_coin_pairs if filter in pair]
        return all_coin_pairs

    def _get_spot_instruments(self):
        publicDataAPI = PublicData.PublicAPI(flag="0")
        return publicDataAPI.get_instruments(instType="SPOT")

    def _get_swap_instruments(self):
        publicDataAPI = PublicData.PublicAPI(flag="0")
        return publicDataAPI.get_instruments(instType="SWAP")

    async def now_ts(self, inst_id: str, bar: str) -> np.int64:
        import datetime
        now = datetime.datetime.now()
        return np.int64(now.timestamp() * 1000)
    

    def convert_ints_to_np_int64(self, obj):
        if isinstance(obj, dict):
            return {k: self.convert_ints_to_np_int64(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_ints_to_np_int64(v) for v in obj]
        elif isinstance(obj, int):
            return np.int64(obj)
        else:
            return obj
    
    async def setup_check_mongodb(self) -> None:
        """
        Set up compound indexes for each collection in the MongoDB database.
        """
        collections = await self.db.list_collection_names()
        print(collections)
        for collection_name in collections:
            collection = self.db[collection_name]
            # Check if the compound index exists, exclude the original index
            list_of_indexes = await collection.list_indexes().to_list(length=None)
            if not any(index['key'] == [("instId", 1), ("timestamp", 1)] for index in list_of_indexes):
                await collection.create_index([("instId", 1), ("timestamp", 1)], unique=True)
                logging.info(f"Created compound index for {collection_name}.")
            else:
                logging.info(f"Compound index for {collection_name} already exists.")

    async def check_existing_data(self, inst_id: str, bar: str) -> Sequence[TIMESTAMP | None]:
        """
        Finds the latest timestamp in the MongoDB collection.
        """
        collection = self.db[f'kline-{bar}']
        pipeline = [
            {
                "$match": {
                    "instId": inst_id,
                    "bar": bar
                }
            },
            {
                "$group": {
                    "_id": None,
                    "start_date": {"$min": "$timestamp"},
                    "end_date": {"$max": "$timestamp"},
                }
            }
        ]

        cursor =  collection.aggregate(pipeline)
        result = await cursor.to_list(length=1)
        if result:
            start_datetime: datetime = result[0]['start_date']
            end_datetime:datetime = result[0]['end_date']
            return np.int64(start_datetime.timestamp() * 1000), np.int64(end_datetime.timestamp() * 1000)
        else:
            return None, None
            
    async def check_missing_data(self, inst_id, bar):
        """
        Checks if there is missing data in the MongoDB collection.
        """
        collection = self.db[f"kline-{bar}"]
        latest_doc = collection.find({'instId': inst_id}, {'timestamp': 1}).sort('timestamp', -1)
        # Get all of them and convert to pd.DataFrame
        df = pd.DataFrame(await latest_doc.to_list(length=None))
        # Check the timeseries is continuous
        return df['timestamp'].diff().dt.total_seconds().dropna().eq(60).all()
    
    async def fix_missing_data(self, inst_id, bar):
        """
        Fixes missing data in the MongoDB collection.
        """
        collection = self.db[f"kline-{bar}"]
        latest_doc = collection.find({'instId': inst_id}, {'timestamp': 1}).sort('timestamp', -1)
        # Get all of them and convert to pd.DataFrame
        df = pd.DataFrame(await latest_doc.to_list(length=None))
        # Check the timeseries is continuous
        if not df['timestamp'].diff().dt.total_seconds().dropna().eq(60).all():
            # Get the missing timestamps
            missing_ts = pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq='1min').difference(df['timestamp'].to_list())
            # Create a new DataFrame with the missing timestamps
            missing_df = pd.DataFrame({'timestamp': missing_ts})
            # Insert the missing data into the collection
            await insert_data_to_mongodb(self.db[f"kline-{bar}"], missing_df)
            logging.info(f"Inserted {len(missing_df)} missing records into {inst_id} {bar}.")

    async def update_early_missing(self, inst_id : str, bar : str, exist_earliest : TIMESTAMP) -> None:
        a = exist_earliest
        b : np.int64 = a
        time_interval= np.int64(0)
        is_first_time = True
        # Fetch until no more data is returned
        while b:
            try:
                params = {
                    'instId': inst_id,
                    'before': "" if is_first_time else str(b),
                    'after': str(a),
                    'bar': bar,
                }
                
                async with self.semaphore:
                    if self.session != None:
                        async with self.session.get(self.market_url, params=params, headers=self.headers) as response:
                               
                                if response.status == 200:
                                    result = await response.json()
                                    if not result['data']:
                                        logging.info(f"No more data to fetch or empty data returned for {inst_id}-{bar}.")
                                        return None
                                    else:
                                        await self.cvt_dic_to_df_insert(f"kline-{bar}", result['data'],bar=bar, inst_id=inst_id)
                                        a = np.int64(result['data'][-1][0]) - np.int64(1)
                                        
                                        if is_first_time:
                                            time_interval: np.int64 = abs(np.int64(result['data'][0][0]) - a)
                                            is_first_time = False
                                        b = a - time_interval - np.int64(4) + np.int64(random.randint(1, 10)*2)
                        
                                elif response.status == 429:
                                    logging.debug(f"Too many requests for {bar} - {inst_id}.")
                                    
                                else:
                                    logging.error(f"Failed to fetch data with status code {response.status}")
                                    return None
            
            except Exception as e:
                logging.error(f"Error occurred: {e}, Retrying...")
                await asyncio.sleep(0.1)
    
    async def update_latest_missing(self) -> None:
        pass
    
    async def cvt_dic_to_df_insert(self, col_name: str, data: dict, bar: str, inst_id: str) -> None:
        """Convert reponse data to pd.Dataframe and insert many into MongoDB

        Args:
            collection_name (str): _description_
            data (dict): _description_
            bar (str): _description_
            inst_id (str): _description_
        """
        col = self.db[col_name]
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote', 'confirm'])
        df['timestamp'] = pd.to_datetime(df['timestamp'].values.astype(np.int64), unit='ms', utc=True).tz_convert('Asia/Shanghai')
        numeric_fields = ['open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote', 'confirm']
        for field in numeric_fields:
            df[field] = pd.to_numeric(df[field], errors='coerce')
        df['instId'] = inst_id
        df['bar'] = bar
        df_dict = df.to_dict('records')
        await col.insert_many(df_dict) # type: ignore
        logging.info(f"Inserted {len(df_dict)} {inst_id} new records into {col_name} asynchronously.")
    
    async def full_kline_updater(self, inst_id: str, bar: str, sleep_time: int = 1, limit: int = 100):
        latest_ts = await self.now_ts(inst_id, bar)
        a = latest_ts
        b : np.int64 = a
        time_interval= np.int64(0)
        is_first_time = True
        # Fetch until no more data is returned
        while b:
            try:
                params = {
                    'instId': inst_id,
                    'before': "" if is_first_time else str(b),
                    'after': str(a),
                    'bar': bar,
                    'limit': str(limit)
                }
                
                async with self.semaphore:
                    if self.session != None:
                        async with self.session.get(self.market_url, params=params, headers=self.headers) as response:
                               
                                if response.status == 200:
                                    result = await response.json()
                                    if not result['data']:
                                        logging.info(f"No more data to fetch or empty data returned for {inst_id}-{bar}.")
                                        return None
                                    else:
                                        await self.cvt_dic_to_df_insert(f"kline-{bar}", result['data'],bar=bar, inst_id=inst_id)
                                        a = np.int64(result['data'][-1][0]) - np.int64(1)
                                        
                                        if is_first_time:
                                            time_interval: np.int64 = abs(np.int64(result['data'][0][0]) - a)
                                            is_first_time = False
                                        b = a - time_interval - np.int64(4) + np.int64(random.randint(1, 10)*2)
                        
                                elif response.status == 429:
                                    logging.debug(f"Too many requests for {bar} - {inst_id}.")
                                    
                                else:
                                    logging.error(f"Failed to fetch data with status code {response.status}")
                                    return None
            
            except Exception as e:
                logging.error(f"Error occurred: {e}, Retrying...")
                await asyncio.sleep(sleep_time)

    async def fetch_in_between(self, inst_id: str, bar: str, itv_earliest: TIMESTAMP, itv_latest: TIMESTAMP) -> None:
        a = itv_latest
        b : np.int64 = a
        time_interval= np.int64(0)
        is_first_time = True
        while b > itv_earliest:
            try:
                params = {
                    'instId': inst_id,
                    'before': "" if is_first_time else str(b),
                    'after': str(a),
                    'bar': bar,
                }
                
                async with self.semaphore:
                    if self.session != None:
                        async with self.session.get(self.market_url, params=params, headers=self.headers) as response:
                                if response.status == 200:
                                    result = await response.json()
                                    if not result['data']:
                                        logging.info(f"No more data to fetch or empty data returned for {inst_id}-{bar}.")
                                        return None
                                    else:
                                        await self.cvt_dic_to_df_insert(f"kline-{bar}", result['data'],bar=bar, inst_id=inst_id)
                                        a = np.int64(result['data'][-1][0]) - np.int64(1)
                                        
                                        if is_first_time:
                                            time_interval: np.int64 = abs(np.int64(result['data'][0][0]) - a)
                                            is_first_time = False
                                        b = a - time_interval - np.int64(4) + np.int64(random.randint(1, 10)*2)
                        
                                elif response.status == 429:
                                    logging.debug(f"Too many requests for {bar} - {inst_id}.")
                                    
                                else:
                                    logging.error(f"Failed to fetch data with status code {response.status}")
                                    return None
            except BulkWriteError as e:
                logging.info(f"New data meets the old one for {inst_id} - {bar}")
                return
            
            except Exception as e:
                logging.error(f"Error occurred: {e}, Retrying...")
                await asyncio.sleep(0.1)
    
    async def kine_manager(self, inst_id: str, bar: str, sleep_time: int = 1, limit: int = 100):
        exist_earliest, exist_latest = await self.check_existing_data(inst_id, bar)
        if exist_earliest == None:
            logging.info(f"Existing data not found for {inst_id} {bar}")
            await self.full_kline_updater(inst_id, bar)
        
        else:
            logging.info(f"Found existing data for {inst_id} {bar} from {exist_earliest} to {exist_latest}.")
            latest_ts = await self.now_ts(inst_id, bar)
            await self.update_early_missing(inst_id, bar, exist_earliest)
            await self.fetch_in_between(inst_id, bar, exist_latest, latest_ts)
            


    async def initialize_update(self):
        # List of restaurants could be big, think in promise of plying across the sums as detailed.
        coin_pairs = await self.get_all_coin_pairs(filter="USDT")
        logging.info(f"Fetching data for {len(coin_pairs)} coin pairs.\n Pairs: {coin_pairs}")
        bar_sizes = self.bar_sizes
        tasks = []
        for inst_id in coin_pairs:
            for bar in bar_sizes:
                tasks.append(
                    asyncio.create_task(self.kine_manager(inst_id, bar))
                )
        await asyncio.gather(* tasks)

    async def main(self):
        # await self.drop_db()
        await self.create_collections_and_indexes()
        # await self.setup_check_mongodb()
        await self.start_session()
        await self.initialize_update()
        await self.close_session()

if __name__ == "__main__":
    updater = AsyncCryptoDataUpdater()
    asyncio.run(updater.main())
