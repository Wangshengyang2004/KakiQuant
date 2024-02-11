from db.BaseDataReader import db_downloader

downloader = db_downloader()

downloader.download_data("BTC-USDT-SWAP", "2021-01-01", "2021-01-02")