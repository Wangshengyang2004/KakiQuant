import pandas as pd
import time
import logging
from pymongo import MongoClient
from util import get_client_str
logging.basicConfig(level=logging.INFO)

class BaseDataHandler:
    def __init__(self, db_name, collection_name):
        self.client = MongoClient(get_client_str())
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def insert_data_to_mongodb(self, data):
        # Check if the data is an instance of a pandas DataFrame
        # Log the type confirmation for debugging purposes
        logging.info("Converting DataFrame to a list of dictionaries for insertion.")
        # Convert the DataFrame to a list of dictionaries
        data_dict = data.to_dict('records')
        try:
            # Attempt to insert the list of dictionaries as multiple documents
            self.collection.insert_many(data_dict)
            logging.info(f"Successfully inserted {len(data_dict)} records.")
        except Exception as e:
            logging.error(f"Error inserting data: {e}")


    def fetch_data(self, start_date, end_date):
        # This method should be overridden by subclasses
        raise NotImplementedError("Subclass must implement abstract method")

    @staticmethod
    def process_data(data):
        # Implement common data processing steps if any
        return data
    
    def update_data(self, start_date, end_date):
        data = self.fetch_data(start_date, end_date)
        data = self.process_data(data)
        self.insert_data_to_mongodb(data)
        
    def inspect_data(self):
        # This method should be overridden by subclasses
        raise NotImplementedError("Subclass must implement abstract method")
    
    def check_data(self, start_date, end_date):
        # This method should be overridden by subclasses
        raise NotImplementedError("Subclass must implement abstract method")
    
    def data_description(self):
        # This method should be overridden by subclasses
        raise NotImplementedError("Subclass must implement abstract method")
    
    def data_visualization(self):
        # This method should be overridden by subclasses
        raise NotImplementedError("Subclass must implement abstract method")
    
    def data_analysis(self):
        # This method should be overridden by subclasses
        raise NotImplementedError("Subclass must implement abstract method")
    
    def get_latest_date(self):
        # Get the latest date in the database, for updating purpose
        try:
            latest_date = self.collection.find_one(sort=[("date", -1)])["date"]
            return latest_date
        except:
            logging.info("No data in the database yet")
            return None
    
    def get_earliest_date(self):
        # Get the earliest date in the database, for updating purpose
        try:
            earliest_date = self.collection.find_one(sort=[("date", 1)])["date"]
            return earliest_date
        except:
            logging.info("No data in the database yet")
            return None
