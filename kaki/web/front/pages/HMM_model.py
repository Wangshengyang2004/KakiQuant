import streamlit as st
from PIL import Image
import os
# from HMM import run

# Function to display code
def display_code():
    st.code("""
import matplotlib.pyplot as plt
under construction
# Your code here
    """)
    

# Streamlit page layout
st.title('Interactive Plot Generator')

# Parameters
params=["r = 1.0","hist_len = 200","retrain_gap = 60"]
st.sidebar.header('Parameters')
param1 = st.sidebar.slider('r', min_value=0.0, max_value=2.0, value=1.0)
param2 = st.sidebar.slider('hist_len', min_value=200.0, max_value=300.0, value=250.0)
param3 = st.sidebar.text_input('date of train', '2023-01-01')
param4 = st.sidebar.text_input('input file', 'rev.csv')
#Run
# if st.sidebar.button('Try paramerters'):
#     run(param1,param2,param3,param4)

# Picture selection
st.sidebar.header('Picture Selection')
lis=['training_result', 'parameter_iteration', 'sharpratio_result',
                                                             "Sharpe_Sample_Weighted_Calculation_result","best_param",
                                                             "training_outside","predict","profit_calculation",
                                                             "threshold_adjustment1","thresh_adj2","dynamic_threshold"]
selected_picture = st.sidebar.selectbox('Select a graph', lis)

# Button to display code
if st.sidebar.button('Display Code'):
    display_code()

# Display selected parameters
st.subheader('Selected Parameters:')
st.write(f'r: {param1}')
st.write(f'hist_len: {param2}')
st.write(f'date of train: {param3}')
st.write(f'input file: {param4}')

# Display selected picture
st.subheader('Selected Picture:')
if selected_picture == 'training_result':
    image = Image.open('../assets/training_result.jpg')
    st.image(image,channels="BGR")
    
elif selected_picture == 'parameter_iteration':
    for i in range(1,6):
        image = Image.open('../assets/parameters'+str(i)+'.jpg')
        st.image(image,channels="BGR")
elif selected_picture == 'sharpratio_result':
    image = Image.open('../assets/sharpratio_result.jpg')
    st.image(image,channels="BGR")
elif selected_picture == 'Sharpe_Sample_Weighted_Calculation_result':
    image = Image.open('../assets/Sharpe_Sample_Weighted_Calculation_result.jpg')
    st.image(image,channels="BGR")
elif selected_picture == 'predict':
    image = Image.open('../assets/predict.jpg')
    st.image(image,channels="BGR")
elif selected_picture == 'best_param':
    image = Image.open('../assets/best_param.jpg')
    st.image(image,channels="BGR")
elif selected_picture == 'training_outside':
    image = Image.open('../assets/training_outside.jpg')
    st.image(image,channels="BGR")
elif selected_picture == 'Sharpe_Sample_Weighted_Calculation_result':
    image = Image.open('../assets/Sharpe_Sample_Weighted_Calculation_result.jpg')
    st.image(image,channels="BGR")
elif selected_picture == 'profit_calculation':
    image = Image.open('../assets/profit_calculation.jpg')
    st.image(image,channels="BGR")

elif selected_picture == 'threshold_adjustment1':
    image = Image.open('../assets/threshold_adjustment1.jpg')
    st.image(image,channels="BGR")
elif selected_picture == 'thresh_adj2':
    image = Image.open('../assets/thresh_adj2.jpg')
    st.image(image,channels="BGR")
elif selected_picture == 'dynamic_threshold':
    image = Image.open('../assets/dynamic_threshold.jpg')
    st.image(image,channels="BGR")


