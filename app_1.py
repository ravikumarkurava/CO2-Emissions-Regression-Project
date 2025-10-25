import streamlit as st
import pickle
import numpy as np
import pandas as pd
import json

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("CO2 Emission Predictor")
st.markdown("Predict CO2 emissions based on vehicle features.")

engine_size = st.slider("Engine Size", min_value=0.9, max_value=8.4, step=0.1)
fuel_city = st.slider("Fuel Consumption (City)", min_value=4.2, max_value=30.6, step=0.1)
fuel_hwy = st.slider("Fuel Consumption (Highway)", min_value=4.0, max_value=20.6, step=0.1)
fuel_comb_l = st.slider("Fuel Consumption (Combined) [L/100km]", min_value=4.1, max_value=26.1, step=0.1)
fuel_comb_mpg = st.slider("Fuel Consumption (Combined) [MPG]", min_value=11.0, max_value=69.0, step=0.1)

input_raw_values = np.array([[engine_size, fuel_city, fuel_hwy, fuel_comb_l, fuel_comb_mpg]])
input_scaled_values = scaler.transform(input_raw_values)

(
    engine_size_scaled,
    fuel_city_scaled,
    fuel_hwy_scaled,
    fuel_comb_l_scaled,
    fuel_comb_mpg_scaled
) = input_scaled_values[0]


cylinders = st.radio("Number of Cylinders", options=[3, 4, 5, 6, 8, 10, 12, 16], horizontal=True)
transmission = st.radio("Transmission Type", options=['A', 'AM', 'AS', 'AV', 'M'], horizontal=True)
fuel_type = st.radio("Fuel Type", options=['D', 'E', 'N', 'X', 'Z'], horizontal=True)

cylinder_options = [3, 4, 5, 6, 8, 10, 12, 16]
cylinder_encoded = [1 if cylinders == cyl else 0 for cyl in cylinder_options]

transmission_options = ['A', 'AM', 'AS', 'AV', 'M']
transmission_encoded = [1 if transmission == t else 0 for t in transmission_options]

fuel_options = ['D', 'E', 'N', 'X', 'Z']
fuel_encoded = [1 if fuel_type == f else 0 for f in fuel_options]

with open('make_freq.json', 'r') as f:
    make_map = json.load(f)

with open('model_freq.json', 'r') as f:
    model_map = json.load(f)

with open('vehicle_class_freq.json', 'r') as f:
    vehicle_class_map = json.load(f)

selected_make = st.selectbox("Select Make (Car brand)", options=list(make_map.keys()))
make_freq = make_map[selected_make]

selected_model = st.selectbox("Select Model (Specific model of the car)", options=list(model_map.keys()))
model_freq = model_map[selected_model]

selected_vehicle_class = st.selectbox("Select Vehicle Class (Body type of the car)", options=list(vehicle_class_map.keys()))
vehicle_class_freq = vehicle_class_map[selected_vehicle_class]

features = [
    engine_size_scaled,
    fuel_city_scaled,
    fuel_hwy_scaled,
    fuel_comb_l_scaled,
    fuel_comb_mpg_scaled,
    *cylinder_encoded,
    *transmission_encoded,
    *fuel_encoded,
    make_freq,
    model_freq,
    vehicle_class_freq
]

feature_names = [
    'engine_size', 'fuel_city', 'fuel_hwy', 'fuel_comb_l', 'fuel_comb_mpg',
    'cylinders_3', 'cylinders_4', 'cylinders_5', 'cylinders_6',
    'cylinders_8', 'cylinders_10', 'cylinders_12', 'cylinders_16',
    'transmission_A', 'transmission_AM', 'transmission_AS', 'transmission_AV', 'transmission_M',
    'fuel_type_D', 'fuel_type_E', 'fuel_type_N', 'fuel_type_X', 'fuel_type_Z',
    'make_freq', 'model_freq', 'vehicle_class_freq'
]

feature_df = pd.DataFrame([features], columns=feature_names)

st.subheader("Features Sent to Model")
st.dataframe(feature_df)

if st.button("Predict CO2 Emission"):
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)
    st.success(f"Estimated CO2 Emission: **{prediction[0]:.2f} g/km**")

