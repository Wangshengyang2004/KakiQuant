import time
import datetime

def timestamp_to_datetime(ts):
    pass

def datetime_to_timestamp(ts_date):
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