import streamlit as st

st.page_link("home.py", label="Home", icon="🏠")
st.page_link("pages/view_kline.py", label="Candlestick", icon="1️⃣")
st.page_link("pages/show_db.py", label="DB status", icon="2️⃣")
st.page_link("http://www.google.com", label="Google", icon="🌎")

# markdown
st.markdown('Frontend Demo for KakiQuant')

# 设置网页标题
st.title('KakiQuant, Your personal finance analysis platform built with ML')

# 展示一级标题
st.header('1. 安装')

st.text('和安装其他包一样，安装 streamlit 非常简单，一条命令即可')
code1 = '''pip3 install streamlit'''
st.code(code1, language='bash')


# 展示一级标题
st.header('2. 使用')

# 展示二级标题
st.subheader('2.1 生成 Markdown 文档')

# 纯文本
st.text('导入 streamlit 后，就可以直接使用 st.markdown() 初始化')

# 展示代码，有高亮效果
code2 = '''import streamlit as st
st.markdown('Streamlit Demo')'''
st.code(code2, language='python')

if __name__ == "__main__":
    pass