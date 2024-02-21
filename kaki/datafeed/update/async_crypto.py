import pandas as pd
from datetime import datetime
import logging
import asyncio
import random
import pymongo
import okx.PublicData as PublicData
from motor.motor_asyncio import AsyncIOMotorClient
import time
class AsyncCryptoDataUpdater:
    def __init__(self):
        self.client = AsyncIOMotorClient('mongodb://localhost:27017')
        self.db = self.client.crypto
        self.collection = self.db.crypto_kline
        self.public_data_api = PublicData(flag="0")
        self.market_url = "https://www.okx.com/api/v5/market/history-mark-price-candles"
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

        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", 
                            handlers=[logging.FileHandler("crypto_AIO.log"), logging.StreamHandler()])
    
    async def insert_data_to_mongodb(self, data):
        if not data.empty:
            data_dict = data.to_dict('records')
            await self.collection.insert_many(data_dict)
            logging.info(f"Inserted {len(data_dict)} new records asynchronously.")
    
    async def get_all_coin_pairs(self):
        publicDataAPI = PublicData.PublicAPI(flag="0")
        spot_result = publicDataAPI.get_instruments(instType="SPOT")
        swap_result = publicDataAPI.get_instruments(instType="SWAP")
        spot_list = [i["instId"] for i in spot_result["data"]]
        swap_list = [i["instId"] for i in swap_result["data"]]
        return spot_list + swap_list
    
    async def process_kline(self, data, inst_id, bar):
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote', 'confirm'])
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        numeric_fields = ['open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote', 'confirm']
        for field in numeric_fields:
            df[field] = pd.to_numeric(df[field], errors='coerce')
        df['instId'] = inst_id
        df['bar'] = bar
        return df
    async def fetch_kline_data(self, inst_id: str, bar: str, initial_delay=1):
        # Automatically fetch all data available, from the very beginning to the present
        end_timestamp = datetime.now().timestamp()
        
        request_count = 0  # track requests to manage rate limiting

        # Initially, 'before' is set to end_timestamp to fetch the newest data first and move backwards
        after = str(end_timestamp * 1000)  # API expects milliseconds
        before = None
        is_first_time = True
        while True:
            if request_count >= self.requests_per_limit:
                time.sleep(self.rate_limit_pause)
                request_count = 0  # reset request count after pausing

            params = {
                'instId': inst_id,
                'before': before,
                'after': after,
                'bar': bar,
            }
            response = await self.http_client.get(self.market_url, params=params, headers=self.headers)
            if response.status_code == 200:
                result = response.json()
                if 'data' in result and result['data']:
                    fetched_data = result['data']
                    df = await self.process_kline(fetched_data, inst_id, bar)
                    await self.insert_data_to_mongodb(df)
                    
                    # Update 'before' to the earliest timestamp in the fetched data to get older data in next request
                    earliest_timestamp = int(result['data'][-1][0])
                    if is_first_time:
                        time_interval = int(result['data'][0][0]) - earliest_timestamp
                        is_first_time = False
                    after = earliest_timestamp
                    before = after - time_interval - 4 + random.randint(1, 10)*2
                    
                else:
                    break  # Break the loop if no data is returned
            
            # Too many requests
            elif response.status_code == 429:
                # Sleep for a random time between 1 and 5 seconds
                sleep_time = random.uniform(1, 5)
                logging.info(f"Too many requests. Sleeping for {sleep_time:.2f} seconds.")
                await asyncio.sleep(sleep_time)
            
            else:
                print(f"Failed to fetch data: {response.text}")
                return  # Break the loop on failure

            request_count += 1

        return None

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

if __name__ == "__main__":
    updater = AsyncCryptoDataUpdater()
    asyncio.run(updater.initialize_update())
