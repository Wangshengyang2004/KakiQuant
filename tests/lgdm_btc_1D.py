import pandas as pd
from kakiquant import crypto_data_klines, lgbm_model
# Download crypto data from MongoDB
kk = kakiquant.crypto_data_klines(symbol="BTC-USDT-SWAP", interval="1D", start_date="2021-01-01", end_date="2024-01-02")
# Automatically add technical indicators
kk.add_technical_indicators()
# Get the data
pd = kk.export_data()