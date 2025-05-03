import numpy as np
import pickle
import os
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

def load_models_and_scaler(load_xgb=True, load_lr=True, load_scaler=True):
    """
    Load XGBoost, lr_model.pkl, and scaler.pkl from disk.
    
    Args:
        load_xgb (bool): Whether to load XGBoost model
        load_lr (bool): Whether to load Logistic Regression model
        load_scaler (bool): Whether to load StandardScaler
    
    Returns:
        tuple: (xgb_model, lr_model, scaler) models and preprocessing objects
    """
    xgb_model = None
    lr_model = None
    scaler = None

    try:
        if load_xgb:
            xgb_model = xgb.Booster()
            xgb_model.load_model('xgb_model.json')
            print("XGBoost model loaded successfully")
        
        if load_lr:
            lr_model = joblib.load('lr_model.pkl')
            print("Logistic Regression model loaded successfully")
        
        if load_scaler:
            scaler = joblib.load('scaler.pkl')
            print("Scaler loaded successfully")
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        print("Using placeholder models for demonstration")
        
        # Create placeholder models if loading fails
        if load_xgb and xgb_model is None:
            xgb_model = xgb.XGBClassifier()
        
        if load_lr and lr_model is None:
            # Placeholder LR model
            class PlaceholderLRModel:
                def __init__(self):
                    self.feature_names_in_ = feat_cols
                
                def predict(self, X):
                    # Return random predictions for demo purposes
                    return np.random.choice([0, 1], size=X.shape[0], p=[0.7, 0.3])
                
                def predict_proba(self, X):
                    # Return random probabilities for demo purposes
                    return np.hstack([
                        np.random.uniform(0.5, 1.0, (X.shape[0], 1)),
                        np.random.uniform(0.0, 0.5, (X.shape[0], 1))
                    ])
            
            lr_model = PlaceholderLRModel()
        
        if load_scaler and scaler is None:
            scaler = StandardScaler()
    
    return xgb_model, lr_model, scaler

# Feature columns used in the models
feat_cols = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
    'complaintScore', 'featurePerCharged',
    'Contract_Month-to-month', 'Contract_Two year',
    'PaymentMethod_Bank transfer (automatic)',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check',
    'tenure_binned_1', 'tenure_binned_2', 'tenure_binned_3',
    'MultipleLines_No', 'MultipleLines_Yes',
    'InternetService_DSL', 'InternetService_Fiber optic',
    'OnlineSecurity_No', 'OnlineSecurity_Yes',
    'OnlineBackup_No', 'OnlineBackup_Yes',
    'DeviceProtection_No', 'DeviceProtection_Yes',
    'TechSupport_No', 'TechSupport_Yes',
    'StreamingTV_No', 'StreamingTV_Yes',
    'StreamingMovies_No', 'StreamingMovies_Yes',
    'cluster'
]


kmeans = joblib.load("kmeans.pkl")
nn_churn = joblib.load("nn_forProb.pkl") 
