import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import json
from streamlit_lottie import st_lottie

# Paths to model and encoders


# Load model and encoders
MODEL_PATH = os.path.join('models', 'xgb_classifier_model.pkl')
GENDER_ENCODER_PATH = os.path.join('models', 'gender_encoder.pkl')
HAS_CR_CARD_ENCODER_PATH = os.path.join('models', 'hasCrCard_encoder.pkl')
IS_ACTIVE_MEMBER_ENCODER_PATH = os.path.join('models', 'isActiveMember_encoder.pkl')

# Paths to animations
HEADER_ANIM_PATH = os.path.join('assets', 'customer.json')
BUTTON_ANIM_PATH = os.path.join('assets', 'button_animation.json')
BALLOON_ANIM_PATH = os.path.join('assets', 'balloon_animation.json')

# Load model and encoders


model = joblib.load(MODEL_PATH)
gender_encoder = joblib.load(GENDER_ENCODER_PATH)
hasCrCard_encoder = joblib.load(HAS_CR_CARD_ENCODER_PATH)
isActiveMember_encoder = joblib.load(IS_ACTIVE_MEMBER_ENCODER_PATH)


# Utility: load local Lottie JSON file
def load_lottie_file(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)

# Prediction function
def make_prediction(features: pd.DataFrame) -> int:
    arr = np.array(features).reshape(1, -1)
    return model.predict(arr)[0]

# Main application
def main():
    
    st.set_page_config(page_title='Customer Churn Predictor', page_icon='🧠', layout='centered')

    # Detect device for responsive layout
    if 'device' not in st.session_state:
        st.session_state['device'] = 'mobile' if st.query_params.get('device') == ['mobile'] else 'desktop'

    # Header animation and title
    header_anim = load_lottie_file(HEADER_ANIM_PATH)
    st_lottie(header_anim, speed=1, loop=True, height=250)
    st.markdown("""
        <h1 style='text-align: center; color: #4CAF50;'>🧠 Customer Churn Predictor</h1>
        <h4 style='text-align: center; color: grey;'>Predict customer churn with smart insights!</h4>
        <hr />
    """, unsafe_allow_html=True)

    # Form for user inputs
    with st.form('predict_form'):
        st.subheader('📋 Customer Details')
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
        # Show button hover animation via CSS (if needed) and process prediction
        st.markdown("""
            <style>
            body {
                background-color: #f0f8ff;
            }
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

        # Prepare and scale data
        geo_map = {'France':[0,0,1], 'Spain':[0,1,0], 'Germany':[1,0,0]}
        geo = geo_map[Geography]
        data = {
            'Surname': Surname, 'Age': int(Age), 'Gender': Gender,
            'CreditScore': int(CreditScore), 'Tenure': int(Tenure),
            'Balance': int(Balance), 'NumOfProducts': NumOfProducts,
            'HasCrCard': HasCrCard, 'IsActiveMember': IsActiveMember,
            'EstimatedSalary': int(EstimatedSalary),
            'Geography_France': geo[0], 'Geography_Spain': geo[1], 'Geography_Germany': geo[2]
        }
        df = pd.DataFrame([list(data.values())], columns=list(data.keys()))
        scaler = StandardScaler()
        df[['CreditScore','Age','Balance','EstimatedSalary']] = scaler.fit_transform(
            df[['CreditScore','Age','Balance','EstimatedSalary']]
        )
        df = df.replace(gender_encoder).replace(hasCrCard_encoder).replace(isActiveMember_encoder)

        # Make prediction
        pred = make_prediction(df.drop('Surname', axis=1))
        result_text = '❌ Churn' if pred == 1 else '✅ Not Churn'
        st.success(f"Customer **{Surname}** is predicted to: **{result_text}** 🎯")

        # Balloons animation after prediction
        balloon_anim = load_lottie_file(BALLOON_ANIM_PATH)
        st_lottie(balloon_anim, speed=1, loop=False, height=300)


# Prediction function
def makePrediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()
