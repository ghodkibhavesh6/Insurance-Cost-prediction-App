import streamlit as st
import pickle
import numpy as np

#load the model
model = pickle.load(open("insurance_model.pkl", "rb"))
columns = pickle.load(open("model_columns.pkl", "rb"))

st.title("💰 Insurance Charges Prediction App")

st.write("Enter customer details to predict insurance charges")

#take input from user
age = st.number_input("Age", 18, 65, 30)

sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
children = st.number_input("Children", 0, 5, 0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

sex = 1 if sex == "male" else 0
smoker = 1 if smoker == "yes" else 0

region_map = {
    "southwest": 3,
    "southeast": 2,
    "northwest": 1,
    "northeast": 0
}
region = region_map[region]

#prediction button
if st.button("Predict Charges"):
    input_data = np.array([[age, sex, bmi, children, smoker, region]])
    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Insurance Charges: ₹ {prediction:.2f}")
