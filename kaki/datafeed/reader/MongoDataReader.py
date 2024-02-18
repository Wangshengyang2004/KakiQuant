from dataclasses import dataclass
from pymongo import MongoClient
import logging
from dotenv import load_dotenv
from datetime import datetime

load_dotenv("../config/config.env")

@dataclass
class MongoTimestamp:
    timestamp: datetime

    def mongo_to_python(self) -> str:
        return self.timestamp.strftime('%Y-%m-%d %H:%M:%S')

    @staticmethod
    def python_to_mongo(timestamp_str: str) -> datetime:
        return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')

class MongoDataReader:
    def __init__(self, db_name, collection_name):
        self.client = MongoClient()  # Assume this is correctly configured to connect to your MongoDB
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.range = self.get_collection_date_range()

    def get_data(self, start_date, end_date, fields=None):
        start_date = MongoTimestamp.python_to_mongo(start_date)
        end_date = MongoTimestamp.python_to_mongo(end_date)
        query = {"timestamp": {"$gte": start_date, "$lte": end_date}}

        projection = {}
        if fields == "full":
            projection = {}  # MongoDB returns all fields if projection is empty
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

        cursor = self.collection.find(query, projection)
        return list(cursor)

    def get_collection_date_range(self):
        pipeline = [
            {"$group": {"_id": None, "start_date": {"$min": "$timestamp"}, "end_date": {"$max": "$timestamp"}}}
        ]
        result = list(self.collection.aggregate(pipeline))
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
    reader = MongoDataReader('crypto_db', 'ETH-USDT-1m')
    data = reader.get_data('2023-01-01', '2023-01-31', fields=None)
    print(data)
    print(reader.range)
    # This will return data within the specified range with the default OLHCV fields.
    # The range will be printed as well.