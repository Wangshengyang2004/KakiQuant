import requests
import time
import pandas as pd

class CryptoDataFetcher:
    def __init__(self):
        self.base_url = "https://www.okx.com/api/v5/market/history-mark-price-candles"
        self.rate_limit_pause = 2  # seconds to wait for rate limit reset
        self.requests_per_limit = 10  # max number of requests per rate limit window

    def fetch_mark_price_klines(self, instId, start_date, end_date, bar='1m', limit=100):
        # Convert start_date and end_date to timestamps
        start_timestamp = int(pd.Timestamp(start_date).timestamp())
        end_timestamp = int(pd.Timestamp(end_date).timestamp())
        
        data = []  # to store fetched data
        request_count = 0  # track requests to manage rate limiting

        # Initially, 'before' is set to end_timestamp to fetch the newest data first and move backwards
        before = str(end_timestamp * 1000)  # API expects milliseconds
        
        while True:
            if request_count >= self.requests_per_limit:
                time.sleep(self.rate_limit_pause)
                request_count = 0  # reset request count after pausing

            params = {
                'instId': instId,
                'before': before,
                'bar': bar,
                'limit': str(limit)
            }
            response = requests.get(self.base_url, params=params)
            if response.status_code == 200:
                result = response.json()
                if 'data' in result and result['data']:
                    fetched_data = result['data']
                    data.extend(fetched_data)
                    
                    # Update 'before' to the earliest timestamp in the fetched data to get older data in next request
                    before = fetched_data[-1][0]
                    
                    # Check if we've fetched data beyond the start_date
                    if int(fetched_data[-1][0]) / 1000 < start_timestamp:
                        break
                else:
                    break  # Break the loop if no data is returned
            else:
                print(f"Failed to fetch data: {response.text}")
                break  # Break the loop on failure

            request_count += 1

        # Process and return the data
        # Note: You may want to convert this data into a DataFrame or another format depending on your needs
        return pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volCcy'])

# Example usage
fetcher = CryptoDataFetcher()
instId = 'BTC-USDT-SWAP'
bar = "1D"
data_2019_2021 = fetcher.fetch_mark_price_klines(instId, '2019-01-01', '2021-12-31', bar=bar)
data_2023_2024 = fetcher.fetch_mark_price_klines(instId, '2023-01-01', '2024-12-31', bar=bar)

# Process or save fetched data as needed
