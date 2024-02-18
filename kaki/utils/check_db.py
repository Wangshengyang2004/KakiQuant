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


