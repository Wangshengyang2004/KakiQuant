import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec

from kaki.kkdatac.crypto import get_price, get_pairs

# fetch all data with 1 day resolution
pairs = get_pairs("kline-1D")

# create a dataframe with all pairs
data = get_price(instId=pairs, bar="1D", fields=["open", "high", "low", "close", "instId"])
data.set_index(["instId", "timestamp"])

# get the last 60 days with groups that have a num of timestamps
# that exceeds 60 days
# ugly code
data['timestamp'] = pd.to_datetime(data['timestamp'])
data = data.groupby('instId').filter(lambda x: len(x) >= 60)
# Sort the DataFrame by 'instId' and 'timestamp'
filtered_data = data.sort_values(by=['instId', 'timestamp'])

# Get the last 60 days' groups for each 'instId'
last_60_days_data = filtered_data.groupby('instId').tail(60)


last_60_days_data = last_60_days_data.set_index(["instId", "timestamp"])
last_60_days_data.loc["1INCH-USDT"]

# compute the percent change
percent_change = last_60_days_data.groupby('instId')['close'].pct_change()
last_60_days_data['Daily Return'] = percent_change

# Drop rows containing NaN values
last_60_days_data.dropna(inplace=True)

# compute the correlation matrix
corr_df = last_60_days_data.pivot_table(values='Daily Return', 
                                        index='timestamp', 
                                        columns='instId').corr()

#remove non-significant correlations
corr_df[corr_df<0.7] = 0
np.fill_diagonal(corr_df.values, 0)

# creating the graph of the correlation vals
graph = nx.Graph(corr_df)

# training the model
node2vec = Node2Vec(graph, 
                    dimensions=32,
                    p=1,            
                    q=3,          
                    walk_length=10, 
                    num_walks=600,  
                    workers=4       
                   )


graph = nx.Graph(corr_df)

model = node2vec.fit(window=3,
                     min_count=1, 
                     batch_words=4 
                    )
X = model.wv.vectors

out = model.wv.most_similar('1INCH-USDT')

out = pd.DataFrame(out)
out.columns = ['ticker', 'Similarity(%)']
out["Similarity(%)"] = out["Similarity(%)"] * 100