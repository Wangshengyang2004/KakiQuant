import matplotlib.pyplot as plt
import finplot
import seaborn 
import mplfinance as mpf
import pandas as pd
import seaborn as sns


def plot_kline(data: pd.DataFrame, type='candle', aio=True):
    # Check the row number of the dataframe
    rows = len(data.index)
    if rows <= 150 or aio == True:
        mpf.plot(data, type=type, style='charles', title=f'K-Line Chart:{data.iloc[0].instId}-{data.iloc[0].bar}')
    else:
        while i < len(data):
            savefig_options = {
            'dpi': 300,  # Increase DPI for higher resolution
            'pad_inches': 0.25  # Optional: Padding around the figure
            }
            mpf.plot(data[i:i+150], type='candle', style='charles',
                    title=f'K-Line Chart:{data[i:i+150].iloc[0].instId}-{data[i:i+150].iloc[0].bar}',
                    ylabel='Price', volume = True, mav=(7,12),savefig=savefig_options)
            i += 150
        

def plot_hmap(data: pd.DataFrame):
    pass

def plot_corrhmap(data: pd.DataFrame) -> None:
    # calculate the correlation matrix on the numeric columns
    corr = data.select_dtypes('number').corr()
    # plot the heatmap
    sns.heatmap(corr)
    
if "__name__" == "__main__":
    from kaki.kkdatac.crypto import get_price
    plot_kline(get_price(target='crypto', instId='ETH-USDT-SWAP', bar='1D'))