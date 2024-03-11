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
        
        if start_date is not None:
            start_date = date_to_datetime(start_date)
        if end_date is not None:
            end_date = date_to_datetime(end_date)
        if self.target == "crypto":
            collection = self.db[f"kline-{bar}"]

        if start_date is None and end_date is None:
            query = {
                    "instId": symbol,
                    "bar": bar
                    }
            
        logging.info(query)
        projection = {}
        if fields == "full":
            projection = {}  # MongoDB returns all fields, including the objectId if projection is empty
        elif fields is None:
            projection = {"_id": 0}  # Return all fields except the objectId
        elif fields == "ohlcv":
            projection = {"_id": 0, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1} # Return OLHCV fields only
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
    data = reader.download()
    print(data)
    data.plot(x='timestamp', y='close')
    plt.show()