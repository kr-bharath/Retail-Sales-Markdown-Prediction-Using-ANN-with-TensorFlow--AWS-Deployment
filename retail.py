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

# Function to load model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = load_model("D:/Final Project/Retail-Sales-Markdown-Prediction-Using-ANN-with-TensorFlow--AWS-Deployment/Retail_Sales_Prediction.h5", custom_objects={'LeakyReLU': LeakyReLU})
    with open('scaler.pkl', 'rb') as f:
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
        Type = st.selectbox('Store Type', ['1', '2', '3'])
        Size = st.selectbox('Store Size', sorted(df['Size'].unique()))
        Dept = st.number_input('Department', min_value=1, max_value=100, step=1)
        IsHoliday = st.selectbox('Is Holiday Week', ['0', '1'])
        Temperature = st.number_input('Temperature', min_value=0.0)
        Fuel_Price = st.number_input('Fuel Price', min_value=0.0)
        MarkDown1 = st.number_input('Markdown 1', min_value=0.0)
        MarkDown2 = st.number_input('Markdown 2', min_value=0.0)
        MarkDown3 = st.number_input('Markdown 3', min_value=0.0)

    with col2:
        MarkDown4 = st.number_input('Markdown 4', min_value=0.0)
        MarkDown5 = st.number_input('Markdown 5', min_value=0.0)
        CPI = st.number_input('CPI (Consumer Price Index)', min_value=0.0)
        Unemployment = st.number_input('Unemployment Rate', min_value=0.0)
        Day = st.slider('Day of the Month', min_value=1, max_value=31)
        Week = st.slider('Week of the Year', min_value=1, max_value=52)
        Month = st.slider('Month', min_value=1, max_value=12)
        Year = st.slider('Year', min_value=2010, max_value=2024)
        lag_1_sales = st.number_input('Lag 1 Sales', min_value=0.0)
        lag_1_markdown = st.number_input('Lag 1 Markdown', min_value=0.0)

    col3, col4 = st.columns(2)
    with col3:
        markdown_holiday_interaction = st.number_input('Markdown Holiday Interaction', min_value=0.0)
        markdown_impacted = st.number_input('Markdown Impacted', min_value=0.0)
        days_until_holiday = st.number_input('Days Until Holiday', min_value=0.0)
        pre_holiday_sales_spike = st.number_input('Pre-Holiday Sales Spike', min_value=0.0)

    submit_button = st.form_submit_button('Predict Retail Sales Price')

# Prediction logic
if submit_button:
    # Prepare input data for prediction
    input_data = np.array([[Store, 
                            Type,  # Keep the raw input Type
                            Size, 
                            Dept, 
                            IsHoliday,  # Keep the raw input for IsHoliday
                            Temperature, 
                            Fuel_Price, 
                            MarkDown1, 
                            MarkDown2, 
                            MarkDown3,
                            MarkDown4, 
                            MarkDown5, 
                            CPI, 
                            Unemployment, 
                            Day, 
                            Week, 
                            Month, 
                            Year, 
                            lag_1_sales, 
                            lag_1_markdown,
                            markdown_holiday_interaction, 
                            markdown_impacted, 
                            days_until_holiday, 
                            pre_holiday_sales_spike]])
    
    # Scale input data
    scaled_input = scaler.transform(input_data)

    # Predict using the model
    prediction = model.predict(scaled_input)
    predicted_price = prediction[0][0]

    # Display the prediction
    st.markdown(f'<h4 style="color:#24cd3b;">Predicted Retail Sales Price: ${predicted_price:.2f}</h4>', unsafe_allow_html=True)
    st.balloons()
