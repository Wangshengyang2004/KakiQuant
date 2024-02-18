"""
A wrapper for downloading and maintaining data
"""
from tqdm import tqdm
import time
import okx.PublicData as PublicData
import threading
from queue import Queue
from kaki.datafeed.handler.CryptoDataHandler import CryptoDataHandler
flag = "0"  # 实盘:0 , 模拟盘：1

publicDataAPI = PublicData.PublicAPI(flag=flag)

# 获取交易产品基础信息
SPOT_result = publicDataAPI.get_instruments(
    instType="SPOT"
)

SWAP_result = publicDataAPI.get_instruments(
    instType="SWAP"
)

bar_type = ["1m", "3m", "5m", "15m", "30m", "1H", "4H", "1D", "1W"]
SPOT_LIST = [i["instId"] for i in SPOT_result["data"]]
SWAP_LIST = [i["instId"] for i in SWAP_result["data"]]

def thread_function(queue):
    while not queue.empty():
        task = queue.get()
        try:
            fetch_candlestick_data(**task)
            time.sleep(0.2)
        finally:
            queue.task_done()

def main():
    task_queue = Queue()
    for bar in bar_type:
        for coin in coin_type:
            task = {
                "start_date": "2020-01-01",
                "end_date": "2024-01-13",
                "inst_id": coin,
                "bar": bar,
                "max_retries": 3,
                "initial_delay": 1
            }
            task_queue.put(task)

    threads = []
    for _ in range(10):  # Number of threads
        t = threading.Thread(target=thread_function, args=(task_queue,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

if __name__ == "__main__":
    main()
