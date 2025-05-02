
import pandas as pd
import numpy as np

def load_data():
    """
    Load and merge data from Excel and CSV files, compute complaint score.
    
    Returns:
        pd.DataFrame: Combined dataframe with all customer data
    """
    # TODO: Replace with your actual implementation
    # Example placeholder implementation:
    data = pd.DataFrame({
        'customerID': [f'cust{i:04d}' for i in range(1000)],
        'gender': np.random.choice(['Male', 'Female'], size=1000),
        'SeniorCitizen': np.random.choice([0, 1], size=1000),
        'Partner': np.random.choice(['Yes', 'No'], size=1000),
        'Dependents': np.random.choice(['Yes', 'No'], size=1000),
        'tenure': np.random.randint(1, 72, size=1000),
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
        'complaintCount': np.random.poisson(0.3, size=1000),
        'complaintSeverity': np.random.choice([0, 1, 2, 3], p=[0.7, 0.15, 0.1, 0.05], size=1000)
    })
    
    # Calculate complaint score
    data['complaintScore'] = data['complaintCount'] * (data['complaintSeverity'] + 1)
    
    return data

def process_data(df):
    """
    Perform one-time feature engineering and clustering on the dataset.
    
    Args:
        df (pd.DataFrame): Raw dataframe from load_data
    
    Returns:
        pd.DataFrame: Processed dataframe with engineered features
    """
    # TODO: Replace with your actual implementation
    # Example placeholder implementation:
    processed_df = df.copy()
    
    # Convert categorical to numeric
    binary_map = {'Yes': 1, 'No': 0, 'No internet service': 0, 'No phone service': 0}
    
    # Process binary fields
    binary_cols = ['PhoneService', 'PaperlessBilling', 'Partner', 'Dependents',
                  'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                  'StreamingTV', 'StreamingMovies']
    
    for col in binary_cols:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].map(binary_map)
    
    # Process contract type
    if 'Contract' in processed_df.columns:
        contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
        processed_df['Contract'] = processed_df['Contract'].map(contract_map)
    
    # Process internet service
    if 'InternetService' in processed_df.columns:
        processed_df['hasFiber'] = (processed_df['InternetService'] == 'Fiber optic').astype(int)
        processed_df['hasDSL'] = (processed_df['InternetService'] == 'DSL').astype(int)
        processed_df['hasInternet'] = ((processed_df['hasFiber'] + processed_df['hasDSL']) > 0).astype(int)
    
    # Create derived features
    processed_df['avgMonthlyCharges'] = processed_df['TotalCharges'] / np.maximum(processed_df['tenure'], 1)
    processed_df['serviceCount'] = processed_df[binary_cols].sum(axis=1)
    
    # Create churn flag (placeholder - you'll use your real data)
    processed_df['Churn'] = np.random.choice([0, 1], p=[0.75, 0.25], size=len(processed_df))
    
    return processed_df
