# 此大类负责下载金融数据，对接数据库，调用数据库

### 待完成：
* 1. 完善MongoDB的对接，增加对接数据库的功能， 包括mysql, dolphinDB
* 2. 增加对接金融数据源的功能，包括：
**    2.1. Tushare Pro
**    2.2. AKshare
**    2.3. Binance
* 3. 数据库下载功能

### MongoDB 数据表定义：
#### 加密货币：
MongoDB - crypto_candle:
BTC-USDT-SWAP-1m
...

MongoDB - crypto_index:

MongoDB - crypto_orderbook:
BTC-USDT-SWAP

#### A股:
MongoDB - cn_stock_detail:
000001.SZ-1m
...
600733.SH-1D

MongoDB - cn_stock_info:
index_list
ETF_list
stock_list
...
