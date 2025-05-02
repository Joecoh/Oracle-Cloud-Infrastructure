# ------------------------------------------------------------------------------
# File: main.py
# Author: Joash Elan Cohen
# Email: joashelangovan@gmail.com
# Created: 2025-05-01
# Description: Streamlit web app for predicting customer churn using a trained
#              XGBoost model. Takes customer input and visualizes prediction.
# License: MIT License
# ------------------------------------------------------------------------------

"""
This is the main entry point for the Customer Churn Predictor app.
Users can input customer data to get churn predictions in real time.
"""


import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import json
import requests
from streamlit_lottie import st_lottie

# Utility: Load Lottie animation from URL
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Utility: Load Lottie animation from local file
def load_lottie_file(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)

# Load model and encoders
MODEL_PATH = os.path.join('models', 'xgb_classifier_model.pkl')
GENDER_ENCODER_PATH = os.path.join('models', 'gender_encoder.pkl')
HAS_CR_CARD_ENCODER_PATH = os.path.join('models', 'hasCrCard_encoder.pkl')
IS_ACTIVE_MEMBER_ENCODER_PATH = os.path.join('models', 'isActiveMember_encoder.pkl')

model = joblib.load(MODEL_PATH)
gender_encoder = joblib.load(GENDER_ENCODER_PATH)
hasCrCard_encoder = joblib.load(HAS_CR_CARD_ENCODER_PATH)
isActiveMember_encoder = joblib.load(IS_ACTIVE_MEMBER_ENCODER_PATH)

# Load animations
HEADER_ANIM_PATH = os.path.join('assets', 'customer.json')
BALLOON_ANIM_PATH = os.path.join('assets', 'balloon_animation.json')
header_anim = load_lottie_file(HEADER_ANIM_PATH)
balloon_anim = load_lottie_file(BALLOON_ANIM_PATH)
customer_anim = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_tll0j4bb.json")
brin_anim = load_lottie_file("assets/brain.json")

# Prediction function
def make_prediction(features: pd.DataFrame) -> int:
    arr = np.array(features).reshape(1, -1)
    return model.predict(arr)[0]

# Main application
def main():
    st.set_page_config(page_title='Customer Churn Predictor', page_icon='🤖', layout='centered')

    if 'device' not in st.session_state:
        st.session_state['device'] = 'mobile' if st.query_params.get('device') == ['mobile'] else 'desktop'

    # Header with animation and title
    st_lottie(header_anim, speed=1, loop=True, height=250)
    cols = st.columns([1, 6])
    with cols[0]:
        st_lottie(brin_anim, height=80, width=80)
    
    with cols[1]:
        st.markdown("""
            <div style='display: flex; align-items: center;'>
                <h1 style='color: #4CAF50; margin: 0;'>Customer Churn Predictor</h1>
            </div>
       """, unsafe_allow_html=True)



    # Form for user inputs
    with st.form('predict_form'):
        cols = st.columns([1, 6])
        with cols[0]:
            st_lottie(customer_anim, height=50, width=80)
        with cols[1]:
            st.subheader('Customer Details')

        cols = st.columns(2) if st.session_state['device'] == 'desktop' else (st.container(), st.container())
        col1, col2 = cols[0], cols[1]

        with col1:
            Surname = st.text_input('Customer Name:')
            Age = st.number_input('Age:', min_value=0, max_value=100)
            Gender = st.radio('Gender:', ['Male', 'Female'])
            Geography = st.radio('Geography:', ['France', 'Spain', 'Germany'])
            Tenure = st.selectbox('Tenure (years):', list(range(1, 11)))

        with col2:
            Balance = st.number_input('Balance:', min_value=0, max_value=10_000_000)
            NumOfProducts = st.selectbox('Number of Products:', [1, 2, 3, 4])
            HasCrCard = st.radio('Has Credit Card:', ['Yes', 'No'])
            IsActiveMember = st.radio('Active Member:', ['Yes', 'No'])
            EstimatedSalary = st.number_input('Estimated Salary:', min_value=0, max_value=10_000_000)
            CreditScore = st.number_input('Credit Score:', min_value=0, max_value=1000)

        submitted = st.form_submit_button('🔍 Make Prediction')

    if submitted:
        st.markdown("""
            <style>
            body { background-color: #f0f8ff; }
            .stButton>button:hover {
                background: linear-gradient(90deg, #ff0000, #00ff00, #0000ff);
                background-size: 200% 200%;
                animation: hueShift 3s infinite;
                color: white;
            }
            @keyframes hueShift {
                0% { background-position: 0% 50%; }
                100% { background-position: 100% 50%; }
            }
            </style>
        """, unsafe_allow_html=True)

        # Prepare data
        geo_map = {'France': [0, 0, 1], 'Spain': [0, 1, 0], 'Germany': [1, 0, 0]}
        geo = geo_map[Geography]

        data = {
            'Surname': Surname,
            'Age': int(Age),
            'Gender': gender_encoder.get(Gender, 0),
            'CreditScore': int(CreditScore),
            'Tenure': int(Tenure),
            'Balance': int(Balance),
            'NumOfProducts': NumOfProducts,
            'HasCrCard': hasCrCard_encoder.get(HasCrCard, 0),
            'IsActiveMember': isActiveMember_encoder.get(IsActiveMember, 0),
            'EstimatedSalary': int(EstimatedSalary),
            'Geography_France': geo[0],
            'Geography_Spain': geo[1],
            'Geography_Germany': geo[2]
        }

        df = pd.DataFrame([list(data.values())], columns=list(data.keys()))

        # Make prediction
        pred = make_prediction(df.drop('Surname', axis=1))
        result_text = '❌ Churn' if pred == 1 else '✅ Not Churn'
        st.success(f"Customer **{Surname}** is predicted to: **{result_text}** 🎯")

        # Show celebration animation
        st_lottie(balloon_anim, speed=1, loop=False, height=300)

if __name__ == '__main__':
    main()
