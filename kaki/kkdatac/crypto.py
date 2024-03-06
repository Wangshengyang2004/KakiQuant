from pymongo import MongoClient
from dotenv import load_dotenv
from okx import PublicData
from kaki.datafeed.reader.MongoDataReader import DownloadData
load_dotenv("../config/db.env")

# Get SWAP coin pairs
def get_swap_pairs() -> list:
    pass

# Get SPOT coin pairs
def get_spot_pairs() -> list:
    pass

def get_price(target:str, instId:str, bar:str, start_date=None, end_date=None, fields='full'):
    reader = DownloadData(target)
    df = reader.download(instId, bar, start_date, end_date, fields)
    return df
