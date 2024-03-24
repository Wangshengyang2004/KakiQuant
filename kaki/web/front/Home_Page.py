import streamlit as st

# # Attempt to center the title in the middle of the page
# # This is a workaround and may not be perfect for all screen sizes or resolutions

# # Add some vertical space before the title
# for _ in range(1000):  # Adjust the range to push the title down to the desired level
#     st.empty()

# Custom HTML for a centered title with minimalistic styling

github_docs_url = "https://github.com/Wangshengyang2004/KakiQuant"

title_html = """
<div style="position: relative; text-align: center; margin: 20px 0;">
    <h1 style="color: #FFFFFF;">KakiQuant, Your personal finance analysis platform fueled by AI</h1>
    <div style="width: 100%; height: 4px; background-color: #FFFFFF; margin-top: 0.25em;"></div>
</div>
"""

st.markdown(title_html, unsafe_allow_html=True)

# # Add some vertical space after the title to ensure it's centered
# for _ in range(10):  # Adjust the range based on your needs
#     st.empty()


# st.page_link("Home_Page.py", label="Home")
# st.page_link("pages/Similar_Klines.py", label="Similar Klines")
# st.page_link("http://www.google.com", label="Google")

# Define CSS style for centering links
center_style = """
    <style>
        .centered-links {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 20px;
        }
        .centered-links a {
            margin: 0 10px;
        }
    </style>
"""


info_box_html = """
<div style="display: flex; justify-content: center; gap: 20px; margin-top: 50px;"">
    <div style="background-color: #333; color: #FFF; padding: 20px; border-radius: 10px;">
        <h2>40+ Go</h2>
        <p>Comprehensive dataset with over 40 Go of data</p>
    </div>
    <div style="background-color: #333; color: #FFF; padding: 20px; border-radius: 10px;">
        <h2>User Friendly</h2>
        <p>Simple UI with clear instructions</p>
    </div>
    <div style="background-color: #333; color: #FFF; padding: 20px; border-radius: 10px;">
        <h2>AI-powered</h2>
        <p>Integrated Deep Learning for powerful data insights</p>
    </div>
</div>
"""
# render information boxes
st.markdown(info_box_html, unsafe_allow_html=True)

st.markdown(center_style, unsafe_allow_html=True)
st.markdown('<div class="centered-links">', unsafe_allow_html=True)
st.page_link("Home_Page.py", label="Home")
st.page_link("pages/Similar_Klines.py", label="Similar Klines")
st.page_link("pages/HMM_model.py", label="HMM Model")
st.page_link("pages/GNN_model.py", label="GNN Model")
st.markdown('</div>', unsafe_allow_html=True)

current_date = "24/03/2024"

footer_html = f"""
<div style="color: #464e5f; padding: 10px; position: relative; bottom: 0; width: 100%; text-align: center;">
    <p>Contact us at: <a href="mailto:contact@kakiquant.com">contact@kakiquant.com</a></p>
    <p><a href="{github_docs_url}" target="_blank">Documentation</a></p>
    <p>Last updated: {current_date}</p>
</div>
"""

st.markdown(footer_html, unsafe_allow_html=True)
# if __name__ == "__main__":
#     pass




