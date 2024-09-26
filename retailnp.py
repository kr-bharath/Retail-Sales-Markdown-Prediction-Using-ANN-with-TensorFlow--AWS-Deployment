import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.layers import LeakyReLU # type: ignore

# Set up page configuration
st.set_page_config(page_title="Retail Sales Prediction Using ANN", layout="wide")

# Page title
st.markdown('<h1 style="color:#7433ff;text-align:center;">Retail Sales Prediction Using ANN</h1>', unsafe_allow_html=True)

# Load the data
df = pd.read_csv('D:/Final Project/Retail-Sales-Markdown-Prediction-Using-ANN-with-TensorFlow--AWS-Deployment/retail_sale_predict.csv')

class columns:

    Type=['A','B','C']
    Type_encoded={'A':1,'B':2,'C':3}
    Holiday=['False','True']
    Holiday_encoded={'False':0,'True':1}

# Function to load model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = load_model("D:/Final Project/Retail-Sales-Markdown-Prediction-Using-ANN-with-TensorFlow--AWS-Deployment/Retail_NPmodel.h5", custom_objects={'LeakyReLU': LeakyReLU})
    with open('scaler1.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# Load model and scaler
model, scaler = load_model_and_scaler()

# Input form for sales prediction
st.markdown("<h4 style=color:#ca33ff>Retail Forecast Input:</h4>", unsafe_allow_html=True)

with st.form('price_prediction_form'):
    col1, col2 = st.columns(2)

    with col1:
        Store = st.number_input('Store', min_value=1, max_value=45, step=1)
        Type = st.selectbox('**Type**',columns.Type)
        Size = st.selectbox('Store Size', sorted(df['Size'].unique()))
        Dept = st.number_input('Department', min_value=1, max_value=100, step=1)
        IsHoliday = st.selectbox('**IsHoliday**',columns.Holiday)
        Temperature = st.number_input('Temperature', min_value=0.0)
        Fuel_Price = st.number_input('Fuel Price', min_value=0.0)

    with col2:
        CPI = st.number_input('CPI (Consumer Price Index)', min_value=0.0)
        Unemployment = st.number_input('Unemployment Rate', min_value=0.0)
        Day = st.slider('Day of the Month', min_value=1, max_value=31)
        Week = st.slider('Week of the Year', min_value=1, max_value=52)
        Month = st.slider('Month', min_value=1, max_value=12)
        Year = st.slider('Year', min_value=2010, max_value=2024)

    submit_button = st.form_submit_button('Predict Retail Sales Price')

# Prediction logic
if submit_button:
    IsHoliday=1 if IsHoliday=='Yes' else 0
    Type= 1 if type == 'A' else 2 if type == 'B' else 3
    # Prepare input data for prediction
    input_data = np.array([[Store,Type,Size,Dept,IsHoliday,Temperature,Fuel_Price, 
                            CPI,Unemployment,Day,Week,Month,Year]])

    # Scale input data
    scaled_input = scaler.transform(input_data)

    # Predict using the model
    prediction = model.predict(scaled_input)
    predicted_price = prediction[0][0]

    # Display the prediction
    st.markdown(f'<h4 style="color:#24cd3b;">Predicted Retail Sales Price: ${predicted_price:.2f}</h4>', unsafe_allow_html=True)
    st.balloons()