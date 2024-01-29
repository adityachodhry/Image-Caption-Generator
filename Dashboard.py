# GUI 

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


st.title('Image Caption Dashboard')

image = ()
selected_image = st.selectbox("Enter the URL for image", image)
ticker = st.sidebar.text_input('Enter College Name')
start_date = st.sidebar.date_input('Start Date')
end_date = st.sidebar.date_input('End Date')

st.title('Image Upload and Display')

# Upload an image file
uploaded_file = st.file_uploader("Choose a JPG file", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # You can perform further processing with the image or extract features as needed.
    # For example, you might want to use a machine learning model to analyze the image.

    # If you want to extract features from the image, you can convert it to a NumPy array.
    image_array = np.array(image)

    # Display the extracted features
    st.write("Image Features:")
    st.write(image_array)