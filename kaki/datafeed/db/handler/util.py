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
    if db_type == "dolpindb":
        pass

def get_client_str(db_type="mongodb"):
    if db_type == "mongodb":
        return f"mongodb://{db_ip}:{db_port}/"
    # Add condition for other database types if needed
    if db_type == "mysql":
        pass
    if db_type == "dolpindb":
        pass

def timestamp_to_readable_date(timestamp, format='%Y-%m-%d %H:%M:%S'):
    """
    Convert a Unix timestamp (in seconds or milliseconds) to a readable date string.

    :param timestamp: Unix timestamp in seconds or milliseconds.
    :param format: Format of the output date string.
    :return: Readable date string.
    """
    # Check if timestamp is in milliseconds (length > 10)
    if len(str(timestamp)) > 10:
        timestamp = timestamp / 1000  # Convert to seconds

    return datetime.fromtimestamp(timestamp).strftime(format)


# For crypto data
def get_available_pairs():
    with open("../config/crypto_pairs.json", "r") as f:
        pairs = json.load(f)
    return pairs

def find_and_add_project_root(marker="requirements.txt"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = current_dir
    while True:
        files = os.listdir(root_dir)
        parent_dir = os.path.dirname(root_dir)
        if marker in files:
            break
        elif root_dir == parent_dir:
            raise FileNotFoundError("Project root marker not found.")
        else:
            root_dir = parent_dir
    if root_dir not in sys.path:
        sys.path.append(root_dir)
    
    return root_dir