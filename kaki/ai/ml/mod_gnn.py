import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from kaki.kkdatac.crypto import get_price, get_pairs

def find_similar_crypto_pairs(instId='1INCH-USDT', days=60):
    # Fetch all data with 1 day resolution
    # Assuming 'col' parameter is required and 'kline-1D' is a valid value for it
    pairs = get_pairs(col="kline-1D")

    # Create a dataframe with all pairs
    # Use the 'pairs' variable directly
    data = get_price(instId=pairs, bar="1D", fields=["open", "high", "low", "close", "instId"])
    data.set_index(["instId", "timestamp"])

    # Convert 'timestamp' to datetime and filter data for groups with at least 'days' number of timestamps
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.groupby('instId').filter(lambda x: len(x) >= days)

    # Sort the DataFrame by 'instId' and 'timestamp'
    filtered_data = data.sort_values(by=['instId', 'timestamp'])

    # Get the last 'days' days for each 'instId'
    last_days_data = filtered_data.groupby('instId').tail(days)
    last_days_data = last_days_data.set_index(["instId", "timestamp"])

    # Compute the percent change and add as a new column
    percent_change = last_days_data.groupby('instId')['close'].pct_change()
    last_days_data['Daily Return'] = percent_change

    # Drop rows containing NaN values
    last_days_data.dropna(inplace=True)

    # Compute the correlation matrix
    corr_df = last_days_data.pivot_table(values='Daily Return', index='timestamp', columns='instId').corr()

    # Remove non-significant correlations
    corr_df[corr_df < 0.7] = 0
    np.fill_diagonal(corr_df.values, 0)

    # Creating the graph of the correlation values
    graph = nx.Graph(corr_df)

    # Training the Node2Vec model
    node2vec = Node2Vec(graph, dimensions=32, walk_length=10, num_walks=600, workers=4)
    model = node2vec.fit(window=3, min_count=1, batch_words=4)

    # Find the most similar cryptocurrency pairs to the input pair
    similar_pairs = model.wv.most_similar(instId)
     # Convert the output to a DataFrame and format the similarity values
    similar_pairs_df = pd.DataFrame(similar_pairs, columns=['Ticker', 'Similarity(%)'])
    similar_pairs_df["Similarity(%)"] = similar_pairs_df["Similarity(%)"] * 100

    return similar_pairs_df