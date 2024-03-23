from pymongo import MongoClient
import logging
from kaki.utils.check_date import date_to_datetime
from kaki.utils.check_db import get_client_str
from typing import Optional
import pandas as pd
from pymongo.collection import Collection
import matplotlib.pyplot as plt

class DownloadData:
    def __init__(self, target: str) -> None:
        self.client = MongoClient(get_client_str())
        self.db = self.client[target]
        self.target = target
    
    def download(self, symbol: str = "BTC-USDT-SWAP", bar: str = "1D",
                 start_date: Optional[str | pd.Timestamp] = None, 
                 end_date: Optional[str | pd.Timestamp] = None, fields=None) -> pd.DataFrame:
        query = {}
        if start_date is not None or end_date is not None:
            query["timestamp"] = {}
            if start_date is not None:
                start_date = date_to_datetime(start_date)
                print(start_date)
                query["timestamp"]["$gte"] = start_date
            if end_date is not None:
                end_date = date_to_datetime(end_date)
                print(end_date)
                query["timestamp"]["$lte"] = end_date
        query["instId"] = symbol
        if self.target == "crypto":
            collection = self.db[f"kline-{bar}"]
                     
        logging.info(query)
        projection = {}
        if fields == "full":
            projection = {}  # MongoDB returns all fields, including the objectId if projection is empty
        elif fields is None:
            projection = {"_id": 0}  # Return all fields except the objectId
        elif fields == "ohlcv":
            projection = {"_id": 0, "timestamp": 1, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1} # Return OLHCV fields only
        elif isinstance(fields, list):
            projection = {"_id": 0, "timestamp": 1}
            for field in fields:
                projection[field] = 1
        else:
            raise ValueError("Invalid fields argument. Must be 'full', None, or a list of field names.")
        # print(projection)
        cursor = collection.find(query, projection)
        return pd.DataFrame(list(cursor)).sort_values(by='timestamp', ascending=True)
        
    def describe(self):
        """
        Returns general information about the MongoDB database and collection.
        """
        return [dict(self.db.command("dbstats"), **self.get_collection_info()),
                
        {
            "database": self.db.name,
            "collection_names": self.db.list_collection_names(),
            "collection_count":len(self.db.list_collection_names()),
        }
        ]

    def get_collection_info(self):
        """
        Returns general information about the collection.
        """
        collection_info = {}
        for collection in self.db.list_collection_names():
            collection_info[collection] = self.db[collection].count_documents({})
        return collection_info
    
    def get_crypto_pairs(self, col) -> list:
        col = self.db[col]
        return col.distinct("instId")
    
    def get_collection_date_range(self, collection: Collection, instId: str, bar: str) -> list:
        pipeline = [
            {"$group": {"_id": None, "start_date": {"$min": "$timestamp"}, "end_date": {"$max": "$timestamp"}, "instId": {"$first": "$instId"}, "bar": {"$first": "$bar"}}}
        ]
        result = list(collection.aggregate(pipeline))
        if result:
            start_date = result[0]['start_date']
            end_date = result[0]['end_date']
            return [start_date, end_date]
        else:
            return [None, None]

if __name__ == "__main__":
    reader = DownloadData('crypto')
    data = reader.download(symbol="BTC-USDT-SWAP", bar="1D",start_date="2023-01-01", end_date="2024-01-01", fields=['open','high','low','close'])
    print(data)
    data.plot(x='timestamp', y='close')
    plt.show()