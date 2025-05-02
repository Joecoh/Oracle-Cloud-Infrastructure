# ------------------------------------------------------------------------------
# File: training_model.py
# Author: Joash Elan Cohen
# Email: joashelangovan@gmail.com
# Created: 2025-05-01
# Description: Script to preprocess data, train an XGBoost model, encode features,
#              and save the model and encoders to disk.
# License: MIT License
# ------------------------------------------------------------------------------

"""
This script handles:
- Loading and preprocessing customer churn data
- Encoding categorical variables
- Training an XGBoost classifier
- Saving the trained model and encoders using joblib
"""

import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

class DataPreprocessor:
    def __init__(self, filepath):
        self.filepath = filepath
        try:
            self.df = pd.read_csv(self.filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at {self.filepath}. Please check the file path.")

    def handle_missing_values(self):
        """Handle missing values by dropping rows with missing CreditScore values."""
        self.df.dropna(subset=['CreditScore'], inplace=True)

    def drop_unrelevant_columns(self):
        """Drop columns that are not relevant for prediction."""
        self.df.drop(columns=['Unnamed: 0', 'id', 'CustomerId', 'Surname'], inplace=True)

    def split_data(self):
        """Split the data into training and testing sets."""
        input = self.df.drop('churn', axis=1)
        output = self.df['churn']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(input, output, test_size=0.2, random_state=42)


class FeatureEncoder:
    def __init__(self):
        self.BinaryEncode = {"Gender": {"Male": 1, "Female": 0}}
        self.train_encoded_geo = None

    def binary_encoding(self, df_train, df_test):
        """Perform binary encoding for Gender column."""
        df_train = df_train.replace(self.BinaryEncode)
        df_test = df_test.replace(self.BinaryEncode)
        return df_train, df_test

    def onehot_encoding(self, df_train, df_test):
        """Perform one-hot encoding for Geography column."""
        geo_enc_train = df_train[['Geography']]
        geo_enc_test = df_test[['Geography']]
        self.train_encoded_geo = OneHotEncoder()
        geo_enc_train = pd.DataFrame(self.train_encoded_geo.fit_transform(geo_enc_train).toarray(), columns=self.train_encoded_geo.get_feature_names_out())
        geo_enc_test = pd.DataFrame(self.train_encoded_geo.transform(geo_enc_test).toarray(), columns=self.train_encoded_geo.get_feature_names_out())
        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)
        df_train_enc = pd.concat([df_train, geo_enc_train], axis=1)
        df_train_enc.drop(['Geography'], axis=1, inplace=True)
        df_test_enc = pd.concat([df_test, geo_enc_test], axis=1)
        df_test_enc.drop(['Geography'], axis=1, inplace=True)
        return df_train_enc, df_test_enc

    def save_encoders(self, output_dir):
        """Save binary encoders and one-hot encoder to files."""
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'gender_encoder.pkl'), 'wb') as file:
            pickle.dump(self.BinaryEncode, file)


class FeatureScaler:
    def __init__(self):
        self.ColScale = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
        self.Scaler = StandardScaler()

    def scale_features(self, df_train, df_test):
        """Scale specified features using standardization."""
        df_train[self.ColScale] = self.Scaler.fit_transform(df_train[self.ColScale])
        df_test[self.ColScale] = self.Scaler.transform(df_test[self.ColScale])
        return df_train, df_test


class ModelTrainer:
    def __init__(self):
        self.param_grid_xgb = {'n_estimators': [50, 100],
                               'max_depth': [3, 5],
                               'learning_rate': [0.1, 0.01],
                               'subsample': [0.5, 0.7]}
        self.xgb_classifier_tuned = XGBClassifier(random_state=42)

    def tune_model(self, x_train, y_train):
        """Tune the XGBoost model using grid search."""
        xgb_classifier_tuned = GridSearchCV(self.xgb_classifier_tuned,
                                            param_grid=self.param_grid_xgb,
                                            scoring='accuracy',
                                            cv=5)
        xgb_classifier_tuned.fit(x_train, y_train)
        return xgb_classifier_tuned.best_estimator_


class ModelEvaluator:
    @staticmethod
    def evaluate_model(model, x_test, y_test):
        """Evaluate the model's performance on the test data."""
        y_predict = model.predict(x_test)
        print(classification_report(y_test, y_predict))


class ModelSaver:
    @staticmethod
    def save_model(model, output_dir):
        """Save the trained model to a pickle file."""
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'xgb_classifier_model.pkl'), 'wb') as file:
            pickle.dump(model, file)


def main():
    # Paths
    dataset_path = os.path.join('data', 'data_C.csv')
    output_dir = 'models'

    # Data Preprocessing
    data_processor = DataPreprocessor(dataset_path)
    data_processor.handle_missing_values()
    data_processor.drop_unrelevant_columns()
    data_processor.split_data()

    # Feature Engineering
    feature_encoder = FeatureEncoder()
    x_train_enc, x_test_enc = feature_encoder.binary_encoding(data_processor.x_train, data_processor.x_test)
    x_train_enc, x_test_enc = feature_encoder.onehot_encoding(x_train_enc, x_test_enc)
    feature_encoder.save_encoders(output_dir)

    feature_scaler = FeatureScaler()
    x_train_scaled, x_test_scaled = feature_scaler.scale_features(x_train_enc, x_test_enc)

    # Modelling (XGBoost)
    model_trainer = ModelTrainer()
    xgb_model = model_trainer.tune_model(x_train_scaled, data_processor.y_train)

    # Model Evaluation
    model_evaluator = ModelEvaluator()
    model_evaluator.evaluate_model(xgb_model, x_test_scaled, data_processor.y_test)

    # Save Model
    model_saver = ModelSaver()
    model_saver.save_model(xgb_model, output_dir)


if __name__ == "__main__":
    main()
