import pandas as pd
import os
from pymongo import MongoClient
import logging
from dotenv import load_dotenv
from ..handler.util import get_client_str
# load_dotenv("../config/config.env")

class MongoDBHandler:
    def __init__(self, db_name, collection_name):
        self.client = MongoClient(get_client_str())
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def insert_one(self, data):
        try:
            self.collection.insert_one(data)
        except Exception as e:
            logging.error(f"Error inserting data: {e}")

    def insert_many_from_df(self, df: pd.DataFrame):
        try:
            self.collection.insert_many(df.to_dict('records'))
        except Exception as e:
            logging.error(f"Error inserting DataFrame: {e}")

    def find_one(self):
        try:
            return self.collection.find_one()
        except Exception as e:
            logging.error(f"Error finding data: {e}")
            return None

# Example usage
if __name__ == "__main__":
    db_uri = os.getenv("MONGO_URI")
    handler = MongoDBHandler(db_uri, "test_db", "test_collection")
    data = {"test": "test"}
    handler.insert_one(data)
    print(handler.find_one())
