
import pandas as pd
import numpy as np
import joblib
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import warnings

warnings.filterwarnings('ignore')

def load_data():
    """
    Load and merge data from Excel and CSV files, compute complaint score.
    
    Returns:
        pd.DataFrame: Combined dataframe with all customer data
    """
    try:
        df_main = pd.read_excel("btUTgX.xlsx")
        complaints = pd.read_csv("newdataset2.csv")
        
        kws = [
            "Service Quality",
            "Inconsistent Internet Speed",
            "No Proactive Support",
            "Overcharging",
            "Not Communicated Extra Charges"
        ]

        # compute raw sentiment per complaint
        try:
            analyzer = SentimentIntensityAnalyzer()
            complaints['sentiment_score'] = (
                complaints['complaint']
                .fillna("")
                .apply(lambda txt: analyzer.polarity_scores(txt)['compound'])
            )
        except:
            # If NLTK not available, create random sentiment scores
            print("Warning: NLTK SentimentIntensityAnalyzer not available. Using random sentiment scores.")
            complaints['sentiment_score'] = np.random.uniform(-1, 1, size=len(complaints))

        # group
        grouped = complaints.groupby("customerID")
        sentiment_agg = grouped['sentiment_score'].sum()

        min_s, max_s = sentiment_agg.min(), sentiment_agg.max()
        sentiment_scaled = -((sentiment_agg - max_s) / (max_s - min_s)).rename("complaintScore")

        # combine into summary
        complaint_summary = sentiment_scaled

        # merge & fill
        df_main = df_main.merge(complaint_summary, on="customerID", how="left")
        df_main["complaintScore"] = df_main["complaintScore"].fillna(0.0)

        print(f"Data loaded successfully with {len(df_main)} customer records")
        return df_main
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        print("Using placeholder data for demonstration")
        
        # Create placeholder data
        return pd.DataFrame({
            'customerID': [f'cust{i:04d}' for i in range(1000)],
            'gender': np.random.choice(['Female', 'Male'], size=1000),
            'SeniorCitizen': np.random.choice([0, 1], size=1000),
            'Partner': np.random.choice(['Yes', 'No'], size=1000),
            'Dependents': np.random.choice(['Yes', 'No'], size=1000),
            'tenure': np.random.randint(0, 72, size=1000),
            'PhoneService': np.random.choice(['Yes', 'No'], size=1000),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], size=1000),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], size=1000),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], size=1000),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], size=1000),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], size=1000),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], size=1000),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], size=1000),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], size=1000),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], size=1000),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], size=1000),
            'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 
                                            'Bank transfer (automatic)', 'Credit card (automatic)'], size=1000),
            'MonthlyCharges': np.random.uniform(20, 120, size=1000),
            'TotalCharges': np.random.uniform(100, 8000, size=1000),
            'Churn': np.random.choice(['Yes', 'No'], size=1000, p=[0.25, 0.75]),
            'complaintScore': np.random.uniform(0, 1, size=1000)
        })

def process_data(df):
    """
    Perform one-time feature engineering and clustering on the dataset.
    
    Args:
        df (pd.DataFrame): Raw dataframe from load_data
    
    Returns:
        pd.DataFrame: Processed dataframe with engineered features
    """
    try:
        # Make a copy to avoid modifying the original
        inp = df.copy()
        inp.loc[inp["tenure"] == 0, "TotalCharges"] = 0


        # Drop the customerID column if it exists in the input
        if 'customerID' in inp.columns:
            inp_id = inp['customerID'].copy()  # Save IDs for later if needed
            inp = inp.drop('customerID', axis=1)
        
        # Make another copy for some operations
        df_copy = inp.copy()
        
        # Map service-related columns
        mapping_phone = {"No": 0, "Yes": 1}
        mapping_multi = {"No": 0, "No phone service": 0, "Yes": 1}
        mapping_internet = {"No": 0, "DSL": 1, "Fiber optic": 1}
        
        for col, mapping in [("PhoneService", mapping_phone),
                            ("MultipleLines", mapping_multi),
                            ("InternetService", mapping_internet)]:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].map(mapping)
        
        # Map additional service columns
        service_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", 
                        "TechSupport", "StreamingTV", "StreamingMovies"]
        for col in service_cols:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].map({"No": 0, "Yes": 1, "No internet service": 0})
        
        # Create new feature based on service features
        service_sum_cols = [c for c in ["PhoneService", "MultipleLines", "InternetService", 
                               "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
                               "TechSupport", "StreamingTV", "StreamingMovies"] if c in df_copy.columns]
        
        inp["featurePerCharged"] = df_copy[service_sum_cols].sum(axis=1) / np.maximum(df_copy["MonthlyCharges"], 1e-5)
        
        # Map additional categorical columns
        if "gender" in inp.columns:
            inp["gender"] = inp["gender"].map({"Female": 0, "Male": 1}).astype("category")
        if "SeniorCitizen" in inp.columns:
            inp["SeniorCitizen"] = inp["SeniorCitizen"].astype("category")
        
        for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
            if col in inp.columns:
                inp[col] = inp[col].map({"No": 0, "Yes": 1}).astype("category")
        
        # Convert numeric columns and map churn
        numeric_cols = [c for c in ["MonthlyCharges", "TotalCharges"] if c in inp.columns]
        if numeric_cols:
            inp[numeric_cols] = inp[numeric_cols].astype("float32")
        
        if "Churn" in inp.columns:
            if inp["Churn"].dtype == object:  # If still string format
                inp["Churn"] = inp["Churn"].map({"No": 0, "Yes": 1})
        
        # Bin tenure into categories
        if "tenure" in inp.columns:
            bins = [0, 12, 24, 48, np.inf]
            labels = [0, 1, 2, 3]
            inp["tenure_binned"] = pd.cut(inp["tenure"], bins=bins, labels=labels)
            inp.loc[inp["tenure"] == 0, 'tenure_binned'] = 0
        
        # Create dummy variables for selected categorical columns
        categorical_cols = [col for col in ['Contract', 'PaymentMethod', 'tenure_binned', "MultipleLines", 
                          'InternetService', "OnlineSecurity", "OnlineBackup", 
                          'DeviceProtection', "TechSupport", "StreamingTV", "StreamingMovies"] if col in inp.columns]
        
        if categorical_cols:
            inp = pd.get_dummies(inp, columns=categorical_cols, drop_first=False, dtype="int")
        
        # Drop columns with unwanted keywords
        to_drop = [col for col in inp.columns 
                if ("No internet service" in col or "No phone service" in col or 
                    "InternetService_No" in col or "One year" in col or 
                    "Mailed" in col or "binned_0" in col)]
        
        inp.drop(to_drop, inplace=True, axis=1, errors='ignore')
        
        # Prepare features for clustering
        # cluster_features = ['tenure_binned_1', 'tenure_binned_2', 'tenure_binned_3', 
        #                     'Contract_Month-to-month', 'Contract_Two year', 
        #                     'TechSupport_Yes', 'TechSupport_No', 
        #                     'OnlineSecurity_No', 'OnlineSecurity_Yes', 
        #                     "InternetService_DSL", 'InternetService_Fiber optic']
        
        
        
        # Load pre-trained KMeans model and assign clusters


        try:
            kmeans = joblib.load('kmeans.pkl')
            # ensure we select the exact columns KMeans saw during training
            feat_names = list(kmeans.feature_names_in_)
            missing = [f for f in feat_names if f not in inp.columns]
            if missing:
                raise ValueError(f"Missing clustering features: {missing}")
            # preserve order
            inp_cluster = inp[feat_names].reset_index(drop=True)
            clusters    = kmeans.predict(inp_cluster)
            inp["cluster"] = pd.Categorical(clusters)
        except Exception as e:
            print(f"Error loading KMeans model: {e}")
            print("Assigning random clusters")
            inp["cluster"] = np.random.randint(0, 2, size=inp.shape[0])
       
        
        # Convert remaining columns to int (except float columns)
        for col in inp.drop(columns=["MonthlyCharges", "TotalCharges", "tenure", "featurePerCharged", "complaintScore"], axis=1, errors='ignore').columns:
            inp[col] = inp[col].astype(int)
        
        print(f"Data processed successfully with {len(inp)} rows and {len(inp.columns)} features")
        return inp
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        if 'inp' in locals():
            print(f"Returning partially processed data with {len(inp)} rows")
            return inp
        else:
            print("Returning original data")
            return df
