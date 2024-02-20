import finplot as fplt
from pymongo import MongoClient
from datetime import datetime
import pandas as pd

def get_crypto_kline(collection, instId, bar, start_date=None, end_date=None):
    """
    Fetches k-line data for a given cryptocurrency instrument from MongoDB.
    
    Parameters:
    - collection: A pymongo collection object for querying.
    - instId: The instrument ID to query for.
    - bar: The granularity of the k-line data (e.g., "1D" for daily).
    - symbol: The symbol to query for.
    - start_date: The start date for the query range (inclusive).
    - end_date: The end date for the query range (inclusive).
    
    Returns:
    - A pandas DataFrame containing the k-line data.
    """
    # Convert start_date and end_date to UNIX timestamps in milliseconds
    query_filter = {
        "instId": instId,
        "bar": bar,
    }
    
    if start_date:
        start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        query_filter["timestamp"] = {"$gte": start_timestamp}
    
    if end_date:
        end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
        if "timestamp" in query_filter:
            query_filter["timestamp"]["$lte"] = end_timestamp
        else:
            query_filter["timestamp"] = {"$lte": end_timestamp}
    
    # Perform the query
    cursor = collection.find(query_filter, {'_id': 0})
    # Convert the cursor to a pandas DataFrame
    df = pd.DataFrame(list(cursor))
    
    # Convert timestamps back to readable dates if necessary
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    return df

# Example usage
if __name__ == "__main__":
    client = MongoClient('mongodb://localhost:27017/')
    db = client['crypto']  # Adjust as per your MongoDB setup
    collection = db['crypto_kline']  # Adjust as per your MongoDB setup
    
    instId = "BTC-USDT"
    bar = "3m"
    # start_date = "2021-01-01"
    # end_date = "2021-12-31"
    
    df = get_crypto_kline(collection, instId, bar,)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)


    # Create a candlestick chart
    ax,ax2 = fplt.create_plot('AAPL Candlestick Chart', rows=2)
    fplt.candlestick_ochl(df[['open', 'close', 'high', 'low']], ax=ax)
    fplt.volume_ocv(df[['open', 'close', 'volume']], ax=ax2)

    # Show the plot
    fplt.show()
