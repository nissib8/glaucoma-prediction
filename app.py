import streamlit as st
import pandas as pd
import pickle

# Load saved files
model = pickle.load(open('model.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))
features = pickle.load(open('features.pkl', 'rb'))

st.title("Glaucoma Disease Prediction")

st.write("Enter patient details")

# Take raw inputs
gender = st.selectbox("Gender", ["Male", "Female"])
family_history = st.selectbox("Family History", ["Yes", "No"])
age = st.number_input("Age", 1, 120)
iop = st.number_input("Intraocular Pressure")
cdr = st.number_input("Cup to Disc Ratio")

# Create input dataframe
input_dict = {
    "Age": age,
    "Intraocular Pressure": iop,
    "Cup to Disc Ratio": cdr,
    f"Gender_{gender}": 1,
    f"Family History_{family_history}": 1
}

input_df = pd.DataFrame([input_dict])

# Align columns
input_df = input_df.reindex(columns=features, fill_value=0)

if st.button("Predict"):
    pred = model.predict(input_df)
    result = le.inverse_transform(pred)[0]

    st.success(f"Prediction: {result}")
