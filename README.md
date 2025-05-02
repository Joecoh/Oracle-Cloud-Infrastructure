# Customer Churn Predictor

This project uses a machine learning pipeline with XGBoost to predict whether a customer will churn, based on historical banking data. The model is deployed via a user-friendly Streamlit web app, enhanced with animations for a rich UI experience. This tool aids businesses in identifying high-risk customers and making informed retention decisions.

👉 Try the deployed app here: [Customer Churn Prediction App](https://oracle-nm-ccp.streamlit.app/)

---

## Table of Contents

* [Overview](#overview)
* [Technologies Used](#technologies-used)
* [Features](#features)
* [Project Structure](#project-structure)
* [Setup Instructions](#setup-instructions)
* [Usage](#usage)
* [Model Evaluation](#model-evaluation)
* [License](#license)

---

## Overview

The **Customer Churn Predictor** performs the following:

* Loads and preprocesses banking customer data.
* Encodes categorical variables using binary and one-hot encoding.
* Scales relevant numerical features.
* Trains and tunes an **XGBoost** model.
* Saves the trained model and encoders.
* Provides an interactive Streamlit interface for real-time churn predictions.
* Enhances user experience with Lottie animations and responsive UI.

---

## Technologies Used

* **Python 3.10**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **XGBoost**
* **Joblib / Pickle**
* **Streamlit**
* **Lottie Animations (streamlit-lottie)**

---

## Features

* 🧹 **Data Cleaning & Preprocessing**: Handles missing values, drops irrelevant features.
* 🔄 **Encoding**: Binary and one-hot encoding for categorical variables.
* 📊 **Feature Scaling**: Standardization for select numerical fields.
* 🤖 **XGBoost Modeling**: GridSearchCV for hyperparameter tuning.
* 📈 **Evaluation Metrics**: Precision, recall, F1-score, accuracy.
* 🌐 **Streamlit App**: Responsive layout with animations and real-time predictions.

---

## Project Structure

```
Customer-Churn-Predictor/
├── assets/                          # Lottie animations
│   ├── balloon_animation.json
│   ├── brain.json
│   └── customer.json
├── data/
│   └── data_C.csv                   # Cleaned dataset
├── models/
│   ├── xgb_classifier_model.pkl
│   ├── gender_encoder.pkl
│   ├── hasCrCard_encoder.pkl
│   └── isActiveMember_encoder.pkl
├── src/
│   └── training_model.py           # Modular training pipeline
├── main.py                         # Streamlit app entry point
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### Prerequisites

* Python 3.10+
* Virtualenv or conda (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/joecoh/Oracle-Cloud-Infrastructure.git
cd Customer-Churn-Prediction

# Create and activate virtual environment
python -m venv env
source env/bin/activate   # Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Train the Model

```bash
python src/training_model.py
```

This will save the XGBoost model and required encoders to the `models/` directory.

### 2. Launch the Streamlit App

```bash
streamlit run main.py
```

Visit the provided local URL in your browser to interact with the predictor.

---

## Model Evaluation

| **Metric**        | **Value (XGBoost)** |
| ----------------- | ------------------- |
| Precision (Churn) | 0.76                |
| Recall (Churn)    | 0.55                |
| F1-Score (Churn)  | 0.64                |
| Accuracy          | 86%                 |

> XGBoost was chosen over Random Forest due to better recall and F1-score, which are critical for minimizing missed churn cases.

---

## License

This project is licensed under the [MIT License](LICENSE).

> Created by Joash Elan Cohen — [joashelangovan@gmail.com](mailto:joashelangovan@gmail.com)
