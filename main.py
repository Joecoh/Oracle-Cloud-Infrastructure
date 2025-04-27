import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
<<<<<<< HEAD
import json
from streamlit_lottie import st_lottie

# Paths to model and encoders
=======

# Load model and encoders
>>>>>>> c6964b6d4e921e3fb5848f58dca0e730dceb911a
MODEL_PATH = os.path.join('models', 'xgb_classifier_model.pkl')
GENDER_ENCODER_PATH = os.path.join('models', 'gender_encoder.pkl')
HAS_CR_CARD_ENCODER_PATH = os.path.join('models', 'hasCrCard_encoder.pkl')
IS_ACTIVE_MEMBER_ENCODER_PATH = os.path.join('models', 'isActiveMember_encoder.pkl')
<<<<<<< HEAD
# Paths to animations
HEADER_ANIM_PATH = os.path.join('assets', 'customer.json')
BUTTON_ANIM_PATH = os.path.join('assets', 'button_animation.json')
BALLOON_ANIM_PATH = os.path.join('assets', 'balloon_animation.json')

# Load model and encoders
=======

>>>>>>> c6964b6d4e921e3fb5848f58dca0e730dceb911a
model = joblib.load(MODEL_PATH)
gender_encoder = joblib.load(GENDER_ENCODER_PATH)
hasCrCard_encoder = joblib.load(HAS_CR_CARD_ENCODER_PATH)
isActiveMember_encoder = joblib.load(IS_ACTIVE_MEMBER_ENCODER_PATH)

<<<<<<< HEAD
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
=======
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
>>>>>>> c6964b6d4e921e3fb5848f58dca0e730dceb911a

if __name__ == '__main__':
    main()
