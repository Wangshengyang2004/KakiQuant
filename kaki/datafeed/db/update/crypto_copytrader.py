import requests
import logging
import time
from datafeed.db.update.mongo_middleware import MongoDBHandler
import pandas as pd
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("start")


class CopyTrader:
    def __init__(self):
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
        # print(data)
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
    




if __name__ == '__main__':
    copytrader = CopyTrader()
    copytrader.concat_df()
    copytrader.df_to_csv("follower.csv")
    # print(copytrader.follower_info[0])
    # print(copytrader.follower_info[0]["nickname"])
    # print(copytrader.follower_info[0]["uniqueName"])
    # print(copytrader.follower_info[0]["winRatio"])
    # print(copytrader.follower_info[0]["yieldRatio"])
    # print(copytrader.follower_info[0]["instruments"])