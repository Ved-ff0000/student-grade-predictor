import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
df = pd.read_csv('C://Users//Vedit//OneDrive//Desktop//archive//student_data.csv')
df.describe()
df.isnull().sum()

le = LabelEncoder()

for col in df.select_dtypes(include = ['object']).columns:
    df[col] = le.fit_transform(df[col])

df.drop(['address','famsize','Medu','Fedu','Mjob','Fjob','traveltime','famsup','nursery','freetime','goout','Dalc','Walc','reason','guardian','paid','activities','famrel','higher','internet'], axis=1,inplace = True, errors = 'ignore')

corr = df.corr()

plt.figure(figsize=(10,7))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
y = df['G3']
X = df.drop('G3', axis=1)
X_train,X_test,y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_test,y_test))


y_pred = model.predict(X_test)
plt.figure(figsize = (6,6))
plt.scatter(y_test, y_pred)

plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()])
plt.xlabel('Actual G3')
plt.ylabel('Predicted G3')
plt.title('Actual vs Predicted value')
plt.show()

import pickle
with open("model.pkl","wb") as f:
    pickle.dump(model, f)

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