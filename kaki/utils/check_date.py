import datetime
import pandas as pd 
import numpy as np

def today_is(date):
    """Returns True if today is the given date, False otherwise."""
    return date == datetime.date.today()

def today(tushare_format=False, akshare_format=False):
    if tushare_format:
        return datetime.date.today().strftime("%Y%m%d")
    elif akshare_format:
        return datetime.date.today().strftime("%Y-%m-%d")
    return datetime.date.today()

def date_to_datetime(date: str | datetime.date | datetime.datetime | np.datetime64 | pd.Timestamp):
    """Converts a date to a datetime object with time set to 00:00:00."""
    return pd.to_datetime(date)

def mts_to_datetime(ts: int):
    """Converts a timestamp to a datetime object."""
    return pd.to_datetime(ts, unit="ms")

if __name__ == "__main__":
    print(today())
    print(today(tushare_format=True))
    print(date_to_datetime("2021-01-01"))
    print(date_to_datetime("20210101"))