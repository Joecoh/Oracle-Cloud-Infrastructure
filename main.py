import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Load model and encoders
MODEL_PATH = os.path.join('models', 'xgb_classifier_model.pkl')
GENDER_ENCODER_PATH = os.path.join('models', 'gender_encoder.pkl')
HAS_CR_CARD_ENCODER_PATH = os.path.join('models', 'hasCrCard_encoder.pkl')
IS_ACTIVE_MEMBER_ENCODER_PATH = os.path.join('models', 'isActiveMember_encoder.pkl')

model = joblib.load(MODEL_PATH)
gender_encoder = joblib.load(GENDER_ENCODER_PATH)
hasCrCard_encoder = joblib.load(HAS_CR_CARD_ENCODER_PATH)
isActiveMember_encoder = joblib.load(IS_ACTIVE_MEMBER_ENCODER_PATH)

# # ---------- Custom CSS Styling ----------
# def local_css():
#     st.markdown("""
#         <style>
#             /* Background image */
#             .stApp {
#                 background-image: url("CPP.png");
#                 background-size: cover;
#                 background-attachment: fixed;
#             }

#             /* Glass effect container */
#             .main-container {
#                 background: rgba(255, 255, 255, 0.85);
#                 padding: 2rem;
#                 border-radius: 10px;
#                 max-width: 800px;
#                 margin: auto;
#             }

#             /* Title styling */
#             h1 {
#                 color: #2c3e50;
#                 font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#                 text-align: center;
#                 margin-bottom: 2rem;
#             }

#             /* Input labels */
#             label {
#                 font-weight: 600;
#                 color: #2c3e50;
#             }
#         </style>
#     """, unsafe_allow_html=True)

# ---------- Main UI ----------
def main():
    # local_css()

    st.markdown("<h1>🧠 Customer Churn Predictor</h1>", unsafe_allow_html=True)

    # Inputs
    Surname = st.text_input("Surname:")
    Age = st.number_input("Age:", 0, 100)
    Gender = st.radio("Gender:", ["Male", "Female"])
    Geography = st.radio("Geography:", ['France', 'Spain', 'Germany'])
    Tenure = st.selectbox("Tenure:", list(range(1, 11)))
    Balance = st.number_input("Balance:", 0, 10000000)
    NumOfProducts = st.selectbox("Number of Products:", [1, 2, 3, 4])
    HasCrCard = st.radio("Has Credit Card:", ["Yes", "No"])
    IsActiveMember = st.radio("Active Member:", ["Yes", "No"])
    EstimatedSalary = st.number_input("Estimated Salary:", 0, 10000000)
    CreditScore = st.number_input("Credit Score:", 0, 1000)

    # Encode geography
    geography_encoding = {'France': [0, 0, 1], 'Spain': [0, 1, 0], 'Germany': [1, 0, 0]}
    geo_encoded = geography_encoding[Geography]

    # Prepare dataframe
    data = {
        'Surname': Surname, 'Age': int(Age), 'Gender': Gender, 'CreditScore': int(CreditScore),
        'Tenure': int(Tenure), 'Balance': int(Balance), 'NumOfProducts': NumOfProducts,
        'HasCrCard': HasCrCard, 'IsActiveMember': IsActiveMember, 'EstimatedSalary': int(EstimatedSalary),
        'Geography_France': geo_encoded[0], 'Geography_Spain': geo_encoded[1], 'Geography_Germany': geo_encoded[2]
    }

    df = pd.DataFrame([list(data.values())], columns=[
        'Surname', 'Age', 'Gender', 'CreditScore', 'Tenure',
        'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
        'EstimatedSalary', 'Geography_France', 'Geography_Spain', 'Geography_Germany'
    ])

    # Feature scaling
    scaler = StandardScaler()
    df[['CreditScore', 'Age', 'Balance', 'EstimatedSalary']] = scaler.fit_transform(
        df[['CreditScore', 'Age', 'Balance', 'EstimatedSalary']]
    )

    # Replace with encoded values
    df = df.replace(gender_encoder)
    df = df.replace(hasCrCard_encoder)
    df = df.replace(isActiveMember_encoder)

    # Predict
    if st.button('🔍 Make Prediction'):
        features = df.drop('Surname', axis=1)
        result = makePrediction(features)
        prediction_text = "❌ Churn" if result == 1 else "✅ Not Churn"
        st.success(f"Mr./Mrs. {Surname} is predicted to: **{prediction_text}**")

    st.markdown("</div>", unsafe_allow_html=True)


# Prediction function
def makePrediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()
