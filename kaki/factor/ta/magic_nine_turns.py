import pandas as pd
import numpy as np
from tqdm import tqdm

# Load the dataset
file_path = '~/Desktop/crypto.kline-1D.csv'  # Update this path to your dataset's location
data = pd.read_csv(file_path)

# Function to identify 'Magic Nine Turns' signals
def identify_magic_nine_turns_signals(data):
    data['Buy_Signal'] = False
    for i in tqdm(range(4, len(data) - 9)):
        if all(data['close'].iloc[i + k] > data['close'].iloc[i + k - 4] for k in range(1, 10)):
            data.at[i + 9, 'Buy_Signal'] = True
    return data

# Function to simulate trades based on 'Magic Nine Turns' signals
def simulate_trades(data):
    trades = data[data['Buy_Signal']].copy()
    trades['Buy_Price'] = trades['open'].shift(-1)
    trades['Sell_Price'] = np.nan
    
    for i in tqdm(trades.index):
        idx = list(data.index).index(i)
        sell_window = data.iloc[idx + 1 : min(idx + 6, len(data))]
        trades.at[i, 'Sell_Price'] = sell_window['high'].max()
    
    trades['Profit'] = (trades['Sell_Price'] - trades['Buy_Price']) / trades['Buy_Price']
    return trades

# Main backtesting function
def backtest_strategy(data) -> list:
    results = []
    unique_pairs = data['instId'].unique()

    for pair in unique_pairs:
        pair_data = data[data['instId'] == pair].copy()
        pair_data['timestamp'] = pd.to_datetime(pair_data['timestamp'])
        pair_data.sort_values(by='timestamp', inplace=True)

        pair_data_with_signals = identify_magic_nine_turns_signals(pair_data)
        trades_simulated = simulate_trades(pair_data_with_signals)

        if not trades_simulated.empty:
            results.append({
                'pair': pair,
                'trades': trades_simulated[['timestamp', 'Buy_Price', 'Sell_Price', 'Profit']].dropna()
            })
    
    return results

# Execute the backtest
results = backtest_strategy(data)

# Example to display the results for the first trading pair with trades
if results:
    for i in len(results):
        print(f"Results for {results[i]['pair']}:")
        print(results[i]['trades'])
else:
    print("No trades found across all pairs.")

