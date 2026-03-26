
import streamlit as st
import pickle 
import numpy as np

model = pickle.load(open("model.pkl","rb"))
st.title("Student Grade Predictor")
school = st.selectbox("School (0=GP, 1=MS)", [0,1])
sex = st.selectbox("Sex (0=F, 1=M)", [0,1])
age = st.slider("Age", 15, 22)
Pstatus = st.selectbox("Parent Status (0=T, 1=A)", [0,1])
studytime = st.slider("Study Time", 1, 4)
failures = st.slider("Failures", 0, 4)
schoolsup = st.selectbox("School Support (0/1)", [0,1])
romantic = st.selectbox("Romantic (0/1)", [0,1])
health = st.slider("Health", 1, 5)
absences = st.slider("Absences", 0, 50)
G1 = st.slider("G1", 0, 20)
G2 = st.slider("G2", 0, 20)

if st.button("Predict"):
    features = np.array([[
        school, sex, age, Pstatus,
        studytime, failures, schoolsup,
        romantic, health, absences,
        G1, G2
    ]])
    prediction = model.predict(features)

    st.success(f"Predicted Final Grade : {prediction[0]:.2f}")