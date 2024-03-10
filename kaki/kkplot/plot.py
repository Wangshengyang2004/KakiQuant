import matplotlib.pyplot as plt
import finplot as fplt
import mplfinance as mpf
import pandas as pd
import seaborn as sns


def plot_kline(df: pd.DataFrame, type='candle', aio=True, method="mpf"):
    data = df.set_index('timestamp')
    if method == "mpf":
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
    elif method == 'finp':
        ax,ax2 = fplt.create_plot('Candlestick Chart', rows=2)
        fplt.candlestick_ochl(data[['open', 'close', 'high', 'low']], ax=ax)
        fplt.volume_ocv(data[["open", "close", "volume"]], ax=ax2)
        fplt.show()

def plot_hmap(data: pd.DataFrame):
    pass

def plot_corrhmap(data: pd.DataFrame) -> None:
    # calculate the correlation matrix on the numeric columns
    corr = data.select_dtypes('number').corr()
    # plot the heatmap
    sns.heatmap(corr)
    
if __name__ == "__main__":
    from kaki.kkdatac.crypto import get_crypto_price
    import time
    t = time.time()
    df = get_crypto_price(instId='BTC-USDT-SWAP', bar='1m')
    print(f'cost:{time.time() - t:.4f}s to read db')
    # plot_kline(df)
    plot_kline(df, method="finp")