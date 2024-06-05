import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn

# Load your trained pipeline
pipe = joblib.load(r"F:\Data Science Bootcamp_OdinSchool\6. Capstone Project\1. ML Capstone_Rachana Gupta_S8237\laptop_price_predictor\pipeline.pkl")

# Load your DataFrame
df = pd.read_csv(r"F:\Data Science Bootcamp_OdinSchool\6. Capstone Project\1. ML Capstone_Rachana Gupta_S8237\laptop_price_predictor\final_dataframe.csv")

# Function to predict laptop price
def predict_price(Company, TypeName, Ram, Weight, Touchscreen, Ips, ppi, Cpu_brand, HDD, SSD, Gpu_brand, OS):
    input_data = pd.DataFrame({
        'Company': [Company],
        'TypeName': [TypeName],
        'Ram': [int(Ram)],
        'Weight': [float(Weight)],
        'Touchscreen': [Touchscreen],
        'Ips': [Ips],
        'ppi': [float(ppi)],
        'Cpu brand': [Cpu_brand],
        'HDD': [int(HDD)],
        'SSD': [int(SSD)],
        'Gpu brand': [Gpu_brand],
        'OS': [OS]
    })

    # Predict log-transformed price
    log_price_prediction = pipe.predict(input_data)[0]
    
    # Convert log-transformed price back to original scale
    price_prediction = np.exp(log_price_prediction)
    
    return round(price_prediction, 2)

# Streamlit app
st.title("Laptop Price Predictor")

# Sidebar with header
st.sidebar.header("Enter the specifications of the laptop for knowing its price estimate")

# Inputs
Company = st.sidebar.selectbox("Company", df["Company"].unique())
TypeName = st.sidebar.selectbox("Type of Laptop", df["TypeName"].unique())
Ram = st.sidebar.selectbox("RAM (GB)", sorted(df["Ram"].unique()))
Weight = st.sidebar.slider("Weight (kg)", float(df["Weight"].min()), float(df["Weight"].max()), float(df["Weight"].mean()))
Touchscreen = st.sidebar.radio("Touchscreen", ["No", "Yes"])
Ips = st.sidebar.radio("IPS Display", ["No", "Yes"])
ppi = st.sidebar.slider("Pixels per Inch (PPI)", float(df["ppi"].min()), float(df["ppi"].max()), float(df["ppi"].mean()))
Cpu_brand = st.sidebar.selectbox("CPU brand", df["Cpu brand"].unique())
HDD = st.sidebar.selectbox("HDD (GB)", sorted(df["HDD"].unique()))
SSD = st.sidebar.selectbox("SSD (GB)", sorted(df["SSD"].unique()))
Gpu_brand = st.sidebar.selectbox("GPU brand", df["Gpu brand"].unique())
OS = st.sidebar.selectbox("OS", df["OS"].unique())

# Convert Yes/No to binary
Touchscreen = 1 if Touchscreen == "Yes" else 0
Ips = 1 if Ips == "Yes" else 0

if st.button("Predict Price"):
    price = predict_price(Company, TypeName, Ram, Weight, Touchscreen, Ips, ppi, Cpu_brand, HDD, SSD, Gpu_brand, OS)
    st.write(f"The predicted price of the laptop is INR {price}")