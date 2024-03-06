import streamlit as st
from pymongo import MongoClient
from kaki.utils import check_db

client = MongoClient(check_db.get_client_str)

# Update the info every 60s
# markdown
st.markdown('Frontend Demo for KakiQuant')

# 设置网页标题
st.title('KakiQuant, Your personal finance analysis platform built with ML')

# 展示一级标题
st.header('1. 安装')

st.text('和安装其他包一样，安装 streamlit 非常简单，一条命令即可')
code1 = '''pip3 install streamlit'''
st.code(code1, language='bash')