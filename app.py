
import streamlit as st
import pickle 
import numpy as np

with open("model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
encoders = data["encoders"]
print(type(data))  

st.title("Student Grade Predictor")
school = st.selectbox("School", encoders['school'].classes_)
sex = st.selectbox("Sex", encoders['sex'].classes_)
age = st.slider("Age", 15, 22)
Pstatus = st.selectbox("Parent Status", encoders['Pstatus'].classes_)
studytime = st.slider("Study Time", 1, 4)
failures = st.slider("Failures", 0, 4)
schoolsup = st.selectbox("School Support", encoders['schoolsup'].classes_)
romantic = st.selectbox("Romantic", encoders['romantic'].classes_)
health = st.slider("Health", 1, 5)
absences = st.slider("Absences", 0, 50)
G1 = st.slider("G1", 0, 20)
G2 = st.slider("G2", 0, 20)

def encode(col, value):
    return encoders[col].transform([value])[0]

school = encode('school', school)
sex = encode('sex', sex)
Pstatus = encode('Pstatus', Pstatus)
schoolsup = encode('schoolsup', schoolsup)
romantic = encode('romantic', romantic)
if st.button("Predict"):
    import pandas as pd

    feature_order = [
        'school', 'sex', 'age', 'Pstatus', 'studytime',
        'failures', 'schoolsup', 'romantic',
        'health', 'absences', 'G1', 'G2'
    ]

    features = pd.DataFrame([[ 
        school, sex, age, Pstatus, studytime,
        failures, schoolsup, romantic,
        health, absences, G1, G2
    ]], columns=feature_order)

    try:
        prediction = model.predict(features)
        st.success(f"Predicted Final Grade : {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Error: {e}")