from pymongo import MongoClient
from dotenv import load_dotenv
from okx import PublicData
from kaki.datafeed.reader.MongoDataReader import DownloadData
import pandas as pd
from typing import Union, List
from tqdm import tqdm
load_dotenv("../config/db.env")

reader = DownloadData("crypto")

# Get SWAP coin pairs
def get_swap_pairs() -> list:
    pass

# Get SPOT coin pairs
def get_spot_pairs() -> list:
    pass

def get_pairs(col:str) -> list:
    return reader.get_crypto_pairs(col)

def get_price(instId: Union[str, List[str]], bar: str, start_date=None, end_date=None, fields=None) -> pd.DataFrame:
    if isinstance(instId, str):
        df = reader.download(instId, bar, start_date, end_date, fields)
    elif isinstance(instId, List):
        dfs = []
        for id_ in tqdm(instId):
            dfs.append(reader.download(id_, bar, start_date, end_date, fields))
        df = pd.concat(dfs).reset_index(drop=True)
    else:
        raise ValueError("instId must be either a string or a list of strings.")
    return df.reset_index(drop=True)

if __name__ == "__main__":
    pairs = get_pairs("kline-1D")
    print(pairs)
    print(get_price("BTC-USDT-SWAP", "1D", "2021-01-01", "2021-01-31"))
    print(get_price(["BTC-USDT-SWAP", "ETH-USDT-SWAP"], "1D", "2021-01-01", "2021-01-31"))
    print(get_price(pairs, "1D"))