from pymongo import MongoClient
import logging
from dotenv import load_dotenv
from kaki.utils.check_date import date_to_datetime
from typing import Union
import pandas as pd
load_dotenv("../config/db.env")

class DownloadData:
    def __init__(self, target: str) -> None:
        self.client = MongoClient()  # Assume this is correctly configured to connect to your MongoDB
        self.db = self.client[target]
        self.target = target
    def download(self, symbol: Union[str, None] = "BTC-USDT-SWAP", bar: str = "1D", start_date:str|None = None, end_date: str|None = None, fields=None):
        if start_date is not None:
            start_date = date_to_datetime(start_date)
        if end_date is not None:
            end_date = date_to_datetime(end_date)
        if self.target == "crypto":
            collection = self.db[f"kline-{bar}"]

        if start_date is None and end_date is None:
            query = {
                    "instId": symbol,
                    "bar": bar}
        print(query)
        projection = {}
        if fields == "full":
            projection = {"_id": 0}  # MongoDB returns all fields if projection is empty
        elif fields is None:
            projection = {"_id": 0, "open": 1, "low": 1, "high": 1, "close": 1, "volume": 1}  # Default OLHCV fields
        elif isinstance(fields, list):
            projection = {"_id": 0}
            for field in fields:
                if field in ["open", "low", "high", "close", "volume"]:  # Assuming these are the only valid fields
                    projection[field] = 1
                else:
                    logging.warning(f"Field '{field}' does not exist in the collection.")
                    raise Exception(f"Field '{field}' does not exist in the collection.")
        else:
            raise ValueError("Invalid fields argument. Must be 'full', None, or a list of field names.")

        cursor = collection.find(query, projection)
        # Return pd.DataFrame
        return pd.DataFrame(list(cursor)).sort_values(by='timestamp', ascending=True)
        
        
    def get_collection_date_range(self, collection):
        pipeline = [
            {"$group": {"_id": None, "start_date": {"$min": "$timestamp"}, "end_date": {"$max": "$timestamp"}}}
        ]
        result = list(collection.aggregate(pipeline))
        if result:
            start_date = result[0]['start_date']
            end_date = result[0]['end_date']
            return [start_date, end_date]
        else:
            return [None, None]

# Example usage:
# reader = MongoDataReader('your_db_name', 'your_collection_name')
# data = reader.get_data('2023-01-01', '2023-01-31', fields=None)
# This will return data within the specified range with the default OLHCV fields.

if __name__ == "__main__":
    reader = DownloadData('crypto')
    data = reader.download(fields="full")
    print(data)
    data.plot(x='timestamp', y='close')