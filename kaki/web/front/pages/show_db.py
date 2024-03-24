import streamlit as st
import pandas as pd
import numpy as np
import datetime 
from datetime import date
# Import the GNN-based function for finding similar crypto pairs
from kaki.ai.ml.mod_gnn import find_similar_crypto_pairs

# Configuration to avoid deprecation warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# Initialize session state variables if they don't exist
if "crypto_pair" not in st.session_state:
    st.session_state.crypto_pair = "BTC-USDT"
if "days" not in st.session_state:
    st.session_state.days = 60
if "similar_pairs" not in st.session_state:
    st.session_state.similar_pairs = None

# Sidebar for user inputs
with st.sidebar:
    st.title("GNN Crypto Analyzer")
    st.session_state.crypto_pair = st.text_input("Enter Crypto Pair", value="BTC-USDT")
    st.session_state.days = st.number_input("Days", min_value=10, max_value=180, value=60)
    analyze_button = st.button("Analyze")

# Main page
st.title("Find Similar Cryptocurrency Pairs using GNN")
st.write("This tool uses Graph Neural Networks (GNN) to analyze and find cryptocurrency pairs similar to your selected pair based on historical trading data.")

# Function to load data and find similar pairs
def analyze_similar_pairs():
    try:
        # Use the imported function to find similar crypto pairs
        similar_pairs_df = find_similar_crypto_pairs(st.session_state.crypto_pair, st.session_state.days)
        st.session_state.similar_pairs = similar_pairs_df
        st.success("Analysis complete!")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Trigger analysis on button click
if analyze_button:
    analyze_similar_pairs()

# Display results
if st.session_state.similar_pairs is not None:
    st.write("## Similar Cryptocurrency Pairs")
    st.dataframe(st.session_state.similar_pairs)

st.write("### How it Works")
st.write("1. Enter the cryptocurrency pair you're interested in (e.g., BTC-USDT).")
st.write("2. Specify the number of days for the analysis period.")
st.write("3. Click the 'Analyze' button to find similar cryptocurrency pairs based on GNN analysis.")
st.write("The analysis leverages historical trading data to identify patterns and relationships between different cryptocurrency pairs using Graph Neural Networks.")
