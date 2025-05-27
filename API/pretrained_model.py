import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import joblib as jb
import pandas as pd
import matplotlib.pyplot as plt
from BACKEND.ML_Model.Customer_segmentation import most_common
from .ML_controller import get_dummies

# Load local Random Forest model instead of MLflow CNN model
CHURN_MODEL_PATH = os.getenv("CHURN_MODEL_PATH", "artifacts/training/model.pkl")
CHURN_SCALER_PATH = os.getenv("CHURN_SCALER_PATH", "artifacts/prepare_base_model/scaler_path.pkl")

try:
    model_churn = jb.load(CHURN_MODEL_PATH)
    scaler_churn = jb.load(CHURN_SCALER_PATH)
except FileNotFoundError as e:
    print(f"Model files not found: {e}")
    model_churn = None
    scaler_churn = None

#____________________PROCESS DATA FUNCTION______________________________________
def process_data_for_churn(df_input):
    df_input.columns = df_input.columns.map(str.strip)
    cols_to_drop = {"Returns", "Age", "Total Purchase Amount", "Churn"}
    df_input.drop(columns=[col for col in cols_to_drop if col in df_input.columns], inplace=True)    
    df_input.dropna(inplace=True)
    if 'Price' not in df_input.columns:
        df_input['Price'] = df_input['Product Price']
    else:
        print("Price column already exists, skipping.") 
    df_input['TotalSpent'] = df_input['Quantity'] * df_input['Price']
    df_features = df_input.groupby("customer_id", as_index=False, sort=False).agg(
        LastPurchaseDate = ("Purchase Date","max"),
        Favoured_Product_Categories = ("Product Category", lambda x: most_common(list(x))),
        Frequency = ("Purchase Date", "count"),
        TotalSpent = ("TotalSpent", "sum"),
        Favoured_Payment_Methods = ("Payment Method", lambda x: most_common(list(x))),
        Customer_Name = ("Customer Name", "first"),
        Customer_Label = ("Customer_Labels", "first"),
    )
    df_features = df_features.drop_duplicates(subset=['Customer_Name'], keep='first')
    df_features['LastPurchaseDate'] = pd.to_datetime(df_features['LastPurchaseDate'])
    df_features['LastPurchaseDate'] = df_features['LastPurchaseDate'].dt.date
    df_features['LastPurchaseDate'] = pd.to_datetime(df_features['LastPurchaseDate'])
    max_LastBuyingDate = df_features["LastPurchaseDate"].max()
    df_features['Recency'] = (max_LastBuyingDate - df_features['LastPurchaseDate']).dt.days
    df_features['LastPurchaseDate'] = df_features['LastPurchaseDate'].dt.date
    df_features['Avg_Spend_Per_Purchase'] = df_features['TotalSpent']/df_features['Frequency'].replace(0,1)
    df_features['Purchase_Consistency'] = df_features['Recency'] / df_features['Frequency'].replace(0, 1)
    df_features.drop(columns=["LastPurchaseDate"],axis=1,inplace=True)
    return df_features

#___________________MODEL FUNCTION______________________________________
def encode_churn(df_features: pd.DataFrame):
    df_copy = df_features.copy()
    df_copy.drop(columns=["customer_id","Customer_Name"],axis=1,inplace=True)
    df_features_encode = get_dummies(df_copy)
    return df_features_encode

def churn_prediction(df_input: pd.DataFrame):
    if model_churn is None or scaler_churn is None:
        raise ValueError("Model or scaler not loaded properly")
    
    df_features = process_data_for_churn(df_input)
    df_features_encode = encode_churn(df_features)
    X = scaler_churn.transform(df_features_encode)  # Use transform instead of fit_transform
    y_pred = model_churn.predict_proba(X)[:, 1]
    df_features['Churn_Probability'] = y_pred
    return df_features

