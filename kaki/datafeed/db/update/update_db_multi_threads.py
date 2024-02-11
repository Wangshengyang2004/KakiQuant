from db_update.crypto_candle import fetch_candlestick_data
from tqdm import tqdm
import time
bar_type = ["1m", "3m", "5m", "15m", "30m", "1H", "4H", "1D", "1W"]
coin_type = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "EOS-USDT-SWAP", "LTC-USDT-SWAP",
              "XRP-USDT-SWAP", "BCH-USDT-SWAP", "BSV-USDT-SWAP", "ETC-USDT-SWAP",
                "TRX-USDT-SWAP", "LINK-USDT-SWAP", "ADA-USDT-SWAP", "DOT-USDT-SWAP",
                  "UNI-USDT-SWAP", "FIL-USDT-SWAP", "XLM-USDT-SWAP", "DOGE-USDT-SWAP",
                    "AAVE-USDT-SWAP", "SUSHI-USDT-SWAP", "YFI-USDT-SWAP", "ATOM-USDT-SWAP",
                      "THETA-USDT-SWAP", "XEM-USDT-SWAP", "COMP-USDT-SWAP", "MKR-USDT-SWAP",
                        "SNX-USDT-SWAP", "XTZ-USDT-SWAP", "ALGO-USDT-SWAP", "NEO-USDT-SWAP",
                          "KSM-USDT-SWAP", "DASH-USDT-SWAP", "ZEC-USDT-SWAP", "EGLD-USDT-SWAP",
                            "ONT-USDT-SWAP", "ZIL-USDT-SWAP", "QTUM-USDT-SWAP", "IOST-USDT-SWAP",
                              "ZRX-USDT-SWAP", "BAT-USDT-SWAP", "ICX-USDT-SWAP", "LSK-USDT-SWAP",
                                "SC-USDT-SWAP", "RVN-USDT-SWAP", "WAVES-USDT-SWAP", "KNC-USDT-SWAP",
                                  "BTT-USDT-SWAP", "CHZ-USDT-SWAP", "ENJ-USDT-SWAP", "MANA-USDT-SWAP",
                                    "STORJ-USDT-SWAP", "OGN-USDT-SWAP", "RLC-USDT-SWAP", "BAL-USDT-SWAP",
                                      "CRV-USDT-SWAP", "SXP-USDT-SWAP", "KAVA-USDT-SWAP", "CTXC-USDT-SWAP",
                                        "BAND-USDT-SWAP", "RLY-USDT-SWAP", ]

import threading
from queue import Queue

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
