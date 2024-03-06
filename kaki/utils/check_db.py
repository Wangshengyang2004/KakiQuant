"""
功能函数，如：获取当前时间，获取当前时间戳，
"""
import pymongo
import json
from dotenv import load_dotenv
import os
from datetime import datetime
import sys
from kaki.utils.check_root_base import find_and_add_project_root

base_dir = os.path.join(find_and_add_project_root(), f"config/db.env")
load_dotenv(base_dir)

db_ip = os.getenv("db_ip")
db_port = os.getenv("db_port")
db_type = os.getenv("db_type")

def get_client(db_type="mongodb"):
    if db_type == "mongodb":
        return pymongo.MongoClient(f"mongodb://{db_ip}:{db_port}/")
    # Add condition for other database types if needed
    if db_type == "mysql":
        pass
    if db_type == "dolphindb":
        pass

def get_client_str(db_type="mongodb"):
    if db_type == "mongodb":
        return f"mongodb://{db_ip}:{db_port}/"
    # Add condition for other database types if needed
    if db_type == "mysql":
        pass
    if db_type == "dolphindb":
        pass

def mongodb_general_info(client) -> dict:
    pass

def get_collection_date_range(collection, instId, bar):
    """
    Returns the earliest (start) and latest (end) timestamps for a specific instId and bar size
    in the specified MongoDB collection.
    
    Parameters:
    - collection: A pymongo collection object.
    - instId: The instrument ID to filter documents by.
    - bar: The bar size to filter documents by.
    
    Returns:
    - A tuple containing the start and end timestamps.
    """
    # Filter documents by instId and bar, then aggregate to find min and max timestamps
    pipeline = [
        {
            "$match": {
                "instId": instId,
                "bar": bar
            }
        },
        {
            "$group": {
                "_id": None,
                "start_date": {"$min": "$timestamp"},
                "end_date": {"$max": "$timestamp"}
            }
        }
    ]
    
    result = list(collection.aggregate(pipeline))
    
    if result:
        start_date = result[0]['start_date']
        end_date = result[0]['end_date']
        return start_date, end_date
    else:
        return None, None
    
    
if __name__ == "__main__":
    client = pymongo.MongoClient('mongodb://192.168.31.120:27017/')  # Adjust as needed
    db = client['crypto']  # Your database name
    collection = db['crypto_kline']  # Your collection name

    instId = "BTC-USDT"  # Example instrument ID
    bar = "1m"  # Example bar size

    start_date, end_date = get_collection_date_range(collection, instId, bar)
    if start_date and end_date:
        print(f"The earliest document for {instId} {bar} is from: {start_date}")
        print(f"The latest document for {instId} {bar} is from: {end_date}")
    else:
        print(f"No documents found in the collection for {instId} {bar}.")