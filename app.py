
import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("model.pkl")

st.title("California House Price Predictor")

# User inputs
MedInc = st.number_input("Median Income")
HouseAge = st.number_input("House Age")
AveRooms = st.number_input("Average Rooms")
AveBedrms = st.number_input("Average Bedrooms")
Population = st.number_input("Population")
AveOccup = st.number_input("Average Occupancy")
Latitude = st.number_input("Latitude")
Longitude = st.number_input("Longitude")

# Predict button
if st.button("Predict"):
    X = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
    prediction = model.predict(X)[0]
    st.success(f"Predicted House Price: ${prediction * 100000:.2f}")
