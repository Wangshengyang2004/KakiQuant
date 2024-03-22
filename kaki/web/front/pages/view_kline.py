import streamlit as st
from pymongo import MongoClient
from kaki.utils.check_db import get_client_str,mongodb_general_info
from kaki.kkplot.plot import plot_corrhmap, plot_kline
client = MongoClient(get_client_str())
collections = [i.split("-")[1] for i in list(mongodb_general_info(client).values())[0]]
print(collections)
# Store the initial value of widgets in session state
# if "visibility" not in st.session_state:
#     st.session_state.visibility = "visible"
#     st.session_state.disabled = False

col1, col2 = st.columns(2)

with col1:
    st.radio(
        "Set the Kline Bar you want to plot",
        key="visibility",
        options=collections,
    )

with col2:
    option = st.selectbox(
        "How would you like to be contacted?",
        ("Email", "Home phone", "Mobile phone"),
        # label_visibility=st.session_state.visibility,
        # disabled=st.session_state.disabled,
    )
