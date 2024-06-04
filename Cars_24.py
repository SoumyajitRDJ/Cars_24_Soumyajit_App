import datetime
import pickle

import pandas as pd
import yfinance as yf
import streamlit as st

st.header('Cars 24 Price Prediction App')

fuel_type = st.selectbox(
    "Select Fuel Type",
    ('Petrol','Diesel','CNG','Electric','LPG'))

col1, col2 = st.columns(2)

with col1:
    engine = st.slider('Set The Engine Power',500,5000,step=100)
with col2:
    Transmission_Type=st.selectbox(
        "Select Transmission Type",
        ('Manual','Automatic'))

Seats = st.selectbox(
    "Select the no of seats",[4,5,6,7,8])

encode_dict={
    'fuel_type':{'Petrol':2,'Diesel':1,'CNG':3,'Electric':4,'LPG':5},
    'Transmission_Type':{'Manual':1,'Automatic':2}
}
def model_pred(Fuel_encoded,Transmission_encoded,Seats,engine):
    with open("car_pred","rb") as file:
        reg_model=pickle.load(file)

        input_features=[[2018.0,1,120000,Fuel_encoded,Transmission_encoded,19.7,engine,46.3,Seats]]
        return reg_model.predict(input_features)

if st.button('Predict'):
    Fuel_encoded = encode_dict['fuel_type'][fuel_type]
    Transmission_encoded = encode_dict['Transmission_Type'][Transmission_Type]
    price=model_pred(Fuel_encoded,Transmission_encoded,Seats,engine)
    st.text("Predicted Price"+str(price))