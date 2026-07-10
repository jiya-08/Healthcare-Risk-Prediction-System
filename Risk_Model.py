import streamlit as st
import pandas as pd
import joblib

model = joblib.load("Risk_Model1.ipynb")

st.title("Healthcare Risk Stratification Model")

# User Inputs
age = st.number_input("Age", min_value=0)

length_of_stay = st.number_input(
    "LengthofStay (days)",
    min_value=0
)

treatment_cost = st.number_input(
    "TreatmentCost",
    min_value=0.0
)

# Prediction
if st.button("Predict Risk"):

    input_data = pd.DataFrame(
        [[age, length_of_stay, treatment_cost]],
        columns=[
            "Age",
            "LengthOfStay",
            "TreatmentCost"
        ]
    )

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    risk = "High Risk" if prediction == 1 else "Low Risk"

    st.success(f"Predicted Risk: {risk}")
    st.write(f"Probability of High Risk: {probability:.2%}")