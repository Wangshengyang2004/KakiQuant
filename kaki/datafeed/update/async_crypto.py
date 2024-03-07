import pandas as pd
from datetime import datetime
import logging
import asyncio
import random
import aiohttp
import okx.PublicData as PublicData
from motor.motor_asyncio import AsyncIOMotorClient
import numpy as np
import typing
import aiohttp
import time

class AsyncCryptoDataUpdater:
    def __init__(self, max_concurrent_requests = 3) -> None:
        self.client = AsyncIOMotorClient('mongodb://localhost:27017')
        self.db = self.client.crypto
        self.market_url = "https://www.okx.com/api/v5/market/history-candles"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "Host": "www.okx.com",
            "Referer": "https://www.okx.com/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
        }
        self.session =  None
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", 
                            handlers=[logging.FileHandler("aiocrypto.log"), logging.StreamHandler()])
    async def drop_db(self):
        await self.client.drop_database('crypto')
        logging.info("Dropped database 'crypto'.")

    async def start_session(self):
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=10)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def close_session(self):
        if self.session and not self.session.closed:
            await self.session.close()

    async def insert_data_to_mongodb(self, collection_name, data: pd.DataFrame) -> None:
        if not data.empty:
            collection = self.db[collection_name]
            data_dict = data.to_dict('records')
            await collection.insert_many(data_dict) # type: ignore
            logging.info(f"Inserted {len(data_dict)} new records into {collection_name} asynchronously.")

    async def get_all_coin_pairs(self):
        # Wrap the synchronous calls in asyncio.to_thread to run them in separate threads
        spot_result = await asyncio.to_thread(self._get_spot_instruments)
        swap_result = await asyncio.to_thread(self._get_swap_instruments)
        
        spot_list = [i["instId"] for i in spot_result["data"]]
        swap_list = [i["instId"] for i in swap_result["data"]]
        return spot_list + swap_list

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
        
    async def check_existing_data(self, inst_id: str, bar: str) -> np.int64:
        """
        Finds the latest timestamp in the MongoDB collection.
        """
        collection = self.db[f"kline-{bar}"]
        latest_doc = collection.find({'instId': inst_id}, {'timestamp': 1}).sort('timestamp', -1).limit(1)
        latest_timestamp = None
        for doc in latest_doc:
            latest_timestamp = doc['timestamp']
        # Convert datetime timestamp to timestamp in milliseconds in np.int64 format
        return np.int64(latest_timestamp.timestamp() * 1000) if latest_timestamp else np.int64(0)

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
        
    async def fetch_kline_data(self, inst_id: str, bar: str, sleep_time: int = 1, limit: int = 100):
        collection_latest = self.check_existing_data(inst_id, bar)
        latest_ts = await self.now_ts(inst_id, bar)
        a = latest_ts
        b : np.int64
        
        is_first_time = True
        # Fetch until no more data is returned
        while True and b > collection_latest:
            try:
                params = {
                    'instId': inst_id,
                    'before': "" if is_first_time else str(b),
                    'after': str(a),
                    'bar': bar,
                    'limit': str(limit)
                }
                
                async with self.semaphore:
                    async with self.session.get(self.market_url, params=params, headers=self.headers) as response:
                            if response.status == 200:
                                result = await response.json()
                                if not result['data']:
                                    logging.info(f"No more data to fetch or empty data returned for {inst_id}-{bar}.")
                                    return None
                                else:
                                    df = pd.DataFrame(result['data'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote', 'confirm'])
                                    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(np.int64), unit='ms')
                                    numeric_fields = ['open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote', 'confirm']
                                    for field in numeric_fields:
                                        df[field] = pd.to_numeric(df[field], errors='coerce')
                                    df['instId'] = inst_id
                                    df['bar'] = bar
                                    await self.insert_data_to_mongodb(f"kline-{bar}", df)  # Adjust as per your actual method signature
                                    logging.info(f"Successfully inserted data for {inst_id} {bar}.")
                                    a = np.int64(result['data'][-1][0]) - np.int64(1)
                                    if is_first_time:
                                        time_interval = abs(np.int64(result['data'][0][0]) - a)
                                        is_first_time = False
                                    b = a - time_interval - np.int64(4) + np.int64(random.randint(1, 10)*2)
                    
                            elif response.status == 429:
                                logging.debug(f"Too many requests for {bar} - {inst_id}.")
                                # await asyncio.sleep(sleep_time/10)
                            else:
                                logging.error(f"Failed to fetch data with status code {response.status}")
                                return None
            
            except Exception as e:
                logging.error(f"Error occurred: {e}, Retrying...")
                await asyncio.sleep(sleep_time)

    async def initialize_update(self):
        # List of restaurants could be big, think in promise of plying across the sums as detailed.
        coin_pairs = await self.get_all_coin_pairs()
        bar_sizes = ['1m', '3m', '5m', '15m', '30m', '1H', '4H', '1D', '1W']
        tasks = []
        for inst_id in coin_pairs:
            for bar in bar_sizes:
                tasks.append(
                    asyncio.create_task(self.fetch_kline_data(inst_id, bar))
                )
        await asyncio.gather(* tasks)

    async def main(self):
        await self.drop_db()
        await self.start_session()
        await self.initialize_update()
        await self.close_session()

if __name__ == "__main__":
    updater = AsyncCryptoDataUpdater()
    asyncio.run(updater.main())
