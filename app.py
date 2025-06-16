import streamlit as st
import pandas as pd
import pickle

# Load model and scaler
with open("model/logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Title
st.title("Titanic Survival Prediction")

# User inputs
st.header("Enter Passenger Details:")
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 25)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare Paid", 0.0, 600.0, 32.2)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Preprocess input
sex = 0 if sex == "male" else 1
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

input_data = pd.DataFrame([{
    "PassengerId": 0,
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked_Q": embarked_Q,
    "Embarked_S": embarked_S
}])

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.success(f"The passenger would have SURVIVED with a probability of {probability:.2f}")
    else:
        st.error(f"The passenger would NOT have survived. Survival probability: {probability:.2f}")
