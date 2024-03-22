import streamlit as st
from pymongo import MongoClient
from kaki.utils import check_db
from kaki.utils.check_db import mongodb_general_info
client = MongoClient(check_db.get_client_str())
# 设置网页标题
st.title('DB Stats')

st.json(f'{mongodb_general_info(client)}')