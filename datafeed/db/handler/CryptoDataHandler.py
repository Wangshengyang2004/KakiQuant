import time
import random
from BaseDataHandler import BaseDataHandler
import okx.MarketData as MarketData
import sys 
import os
from datetime import datetime
import requests

from util import find_and_add_project_root
root_dir = find_and_add_project_root()
# print(root_dir)
sys.path.append(root_dir)
from utils.check_gpu import enable_cudf_acceleration
from utils.check_date import today
import logging 
# Try GPU acceleration
enable_cudf_acceleration()
import pandas as pd


logging.basicConfig(level=logging.INFO)


class CryptoDataHandler(BaseDataHandler):
    def __init__(self, db_name:str ,collection_name:str, inst_id:str, bar:str):
        collection_name = f"{inst_id}-{bar}".replace('/', '-').replace(' ', '')
        super().__init__(db_name, collection_name)
        self.flag = "0"  # Production trading:0 , demo trading:1
        self.api_client = MarketData.MarketAPI(flag=self.flag)
        self.inst_id = inst_id
        self.bar = bar

    def fetch_kline_data(self, start_date, end_date, max_retries:int=3, initial_delay:int=1, save_to_db:bool=True, return_df:bool=False):
        # Convert start and end dates to Unix timestamp in milliseconds
        start_timestamp = int(pd.Timestamp(start_date).timestamp()) * 1000
        end_timestamp = int(pd.Timestamp(end_date).timestamp()) * 1000

        # Initially, 'before' is None to fetch the latest data
        a = end_timestamp
        b = None
        is_First_time = True
        all_data = pd.DataFrame()
        while True:
            retries = 0
            while retries < max_retries:
                try:
                    result = self.api_client.get_history_candlesticks(
                        instId=self.inst_id,
                        before=str(b) if b else "",
                        after=str(a),
                        bar=self.bar
                    )

                    # Check if result is empty or contains data
                    if not result['data']:
                        logging.info("No more data to fetch or empty data returned.")
                        return all_data if return_df else None

                    # Process and insert data to MongoDB
                    df = self.process_kline(result['data'])
                    
                    if save_to_db and isinstance(df, pd.DataFrame):
                        self.insert_data_to_mongodb(df)
                        logging.info("Inserting in the MongoDB Succeed")
                    if return_df:
                        all_data = pd.concat([all_data, df])

                    # Update 'before' and 'after' for the next request
                    earliest_timestamp = int(result['data'][-1][0])
                    latest_timestamp = int(result['data'][0][0])
                    if is_First_time:
                        time_interval = latest_timestamp - earliest_timestamp
                        is_First_time = False
                    a = earliest_timestamp
                    b = a - time_interval - 4 + random.randint(1, 10)*2

                    if a <= start_timestamp:
                        logging.info("Reached the start date.")
                        self.df_csv = all_data
                        del all_data
                        if self.save_kline:
                            self.save_kline() 
                        return self.df_csv if return_df else None

                    break

                except Exception as e:
                    logging.info(f"Error occurred: {e}")
                    retries += 1
                    time.sleep(initial_delay * retries)  # Exponential backoff
                    if retries == max_retries:
                        logging.info("Max retries reached. Exiting.")
                        return all_data if return_df else None
                    logging.info(f"Retrying... Attempt {retries}/{max_retries}")

    def process_kline(self, data):
        # Additional processing specific to crypto data
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote', 'confirm'])
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        df['Date'] = df['timestamp'].dt.strftime('%Y%m%d%H%M%S')
        numeric_fields = ['open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote', 'confirm']
        for field in numeric_fields:
            df[field] = pd.to_numeric(df[field], errors='coerce')
        return df

    def save_kline(self):
        filename = f"{self.inst_id}_{self.bar}_{today(tushare_format=True)}"
        # Get the absolute save path
        save_path = os.path.join(root_dir, "data", "crypto", "kline")
        # Define the full file path
        full_path = os.path.join(save_path, filename)
        # Save the DataFrame to this path
        self.df_csv.to_csv(full_path, index=False)
        logging.info(f"Data saved to {full_path}")


    # fetch Copytrader data
    def fetch_copytrader_data(self):
        pass

    def init_copytrader(self):
        self.current_timestamp = int(time.time())
        self.proxies = {
            "http": "http://localhost:7890",
            "https": "http://localhost:7890",
        }
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        self.HOST = "https://www.okx.com/priapi/v5/ecotrade/public"
        self.size = 20
        self.start = 1
        self.end = 20
        self.follower_list_uri = "follow-rank?size={size}&start={start}&fullState=0&countryId=CN"
        self.follower_list_url_list = self.assemble_url()
        
        logging.info("follower_list_url_list: {}".format(self.follower_list_url_list))
    
    def assemble_url(self):
        url_list = []
        for i in range(self.start, self.end):
            url_list.append(self.HOST + "/" + self.follower_list_uri.format(size=self.size, start=i))
        return url_list
    
    def get_follower_list(self):
        logging.info("Retrieving follower list")
        follower_info = []  # 访问者信息列表
        self.uniqueName_list = []  # 唯一名称列表
        for url in self.follower_list_url_list:  # 遍历访问者URL列表
            data = self._make_request(url)  # 发起请求获取数据
            if data and len(data.get("data", [])) > 0:  # 如果数据存在且不为空
                for user in data["data"][0]["ranks"]:  # 遍历数据中的用户列表
                    nickname = user["nickName"],  # 访问者昵称
                    uniqueName = user["uniqueName"],  # 唯一名称
                    winRatio = user["winRatio"]  # 胜率
                    yieldRatio = user["yieldRatio"]  # 收益率
                    instruments = user["instruments"]  # 交易工具列表
                    self.uniqueName_list.append(uniqueName)  # 将唯一名称添加到唯一名称列表
                    follower_info.append({  # 将访客信息添加到访客信息列表
                        "nickname": nickname,
                        "uniqueName": uniqueName,
                        "winRatio": winRatio,
                        "yieldRatio": yieldRatio,
                        "instruments": len(instruments)
                    })
            else:
                logging.error("No data received or data is empty")  # 没有收到数据或数据为空
                continue
            time.sleep(0.1)  # 休眠0.1秒
        return follower_info
    
    def get_single_copytrader_info(self, uniqueName):
        """
        Fetches the detailed information of a single copytrader.

        :param uniqueName: str - The unique name identifier for a copytrader.
        :return: dict - A dictionary containing the details of the copytrader.
        """
        uniqueName = uniqueName[0]
        logging.info(f"Fetching info for copytrader: {uniqueName}")
        follower_detail_uri = f"trade-info?t={self.current_timestamp}&uniqueName={uniqueName}"
        url = f"{self.HOST}/{follower_detail_uri}"
        data = self._make_request(url)
        logging.info(data)
        if data:
            try:
                assert len(data) != 0
                user = data["data"][0]
                return {
                    "uniqueName": uniqueName,
                    "shareRatio": user["shareRatio"],
                    "totalFollowerNum": user["totalFollowerNum"],
                    "followerLimit": user["followerLimit"],
                    "followerNum": user["followerNum"]
                }
            except AssertionError:
                logging.error("copytrader_info data is empty for " + uniqueName)
            except Exception as e:
                logging.error(f"Error processing data for {uniqueName}: {e}")
        return None

    def get_all_copytraders_info(self):
        """
        Fetches the detailed information of all copytraders in the uniqueName list.
        
        :return: list - A list of dictionaries, each containing the details of a copytrader.
        """
        all_follower_details = []
        for uniqueName in self.uniqueName_list:
            follower_detail = self.get_single_copytrader_info(uniqueName)
            if follower_detail:
                all_follower_details.append(follower_detail)
            time.sleep(0.1)
        return all_follower_details
    
    def trade_detail(self, uniqueName):
        """
        @param: uniqueName: str
        @return: trade_detail: dict
        包括每一笔交易的详细信息
        """
        
        logging.info("trade_detail")
        self.trade_detail_uri = "trade-detail?t={self.current_timestamp}&uniqueName={uniqueName}"
        url = self.HOST + "/" + self.trade_detail_uri.format(self=self, uniqueName=uniqueName)
        r = requests.get(url, headers=self.headers, proxies=self.proxies)
        if r.status_code!= 200:
            print(r.status_code)
        data = r.json()
        try:
            assert len(data) != 0
            for user in data["data"][0]:
                pass
            time.sleep(0.1)
        except:
            logging.error("data is empty")
    
    # 此函数用于将多个字典融合成DataFrame对象
    def concat_df(self):
        """
        每一行是一个跟单者的全部信息， 包括用户基本信息，跟单信息
        @param: follower_info: dict
        @param: follower_detail: dict
        @return: follower_df: DataFrame
        """
        follower_info = self.get_follower_list()
        follower_detail = self.get_all_copytraders_info()
        follower_df = pd.DataFrame(follower_info)
        follower_detail_df = pd.DataFrame(follower_detail)
        follower_df = pd.merge(follower_df, follower_detail_df, on="uniqueName", how="left")
        if follower_df.empty:
            logging.error("follower_df is empty")
        else:
            self.follower_df = follower_df


    #此函数用于将DataFrame对象转换为CSV格式。
    def df_to_csv(self, filepath: str):
        """
        将DataFrame保存为CSV文件。

        参数:
        df (pandas.DataFrame): 要保存为CSV的DataFrame。
        filepath (str): 要保存的CSV文件路径。
        """
        # 检查df是否为DataFrame类型
        if not isinstance(self.follower_df, pd.DataFrame):
            raise ValueError('输入必须是pandas DataFrame对象。')
        
        # 检查文件路径是否有效
        if not filepath.endswith('.csv'):
            raise ValueError('文件路径必须以.csv结尾。')
        
        # 将DataFrame保存为CSV文件
        self.follower_df.to_csv(filepath, index=False)

    def _make_request(self, url):
        """Helper method to make HTTP requests."""
        try:
            response = requests.get(url, headers=self.headers, proxies=self.proxies)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as http_err:
            logging.error(f"HTTP error occurred: {http_err}")
        except Exception as err:
            logging.error(f"An error occurred: {err}")
        return None

if __name__ == "__main__":
    # Create an instance of the CryptoDataHandler
    crypto_handler = CryptoDataHandler("crypto_db", "crypto_collection", "ETH-USDT-SWAP", "1m")

    # Example to fetch and process data
    start_date = "2023-01-01"
    end_date = "2024-02-01"
    fetched_data = crypto_handler.fetch_kline_data(start_date, end_date, return_df=True, save_to_db=True)

    # Example of how to use fetched data
    if not fetched_data.empty:
        print("Fetched data:")
        # print(fetched_data.head())
        print(fetched_data.describe())
    else:
        print("No data fetched.")
