
import numpy as np
import pandas as pd
from scipy.special import expit

# --- Constants ---
# Feature columns used across retention strategies
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
P = len(feat_cols)

# Names of the six yes/no service pairs
pair_names = [
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies'
]

# Toggle costs for services
toggle_costs = {
    'PhoneService': 17.5,
    'InternetService_DSL': 21.875,
    'InternetService_Fiber optic': 43.75,
    'OnlineSecurity_Yes': 4.375,
    'OnlineBackup_Yes': 4.375,
    'DeviceProtection_Yes': 4.375,
    'TechSupport_Yes': 4.375,
    'StreamingTV_Yes': 8.75,
    'StreamingMovies_Yes': 8.75,
}

# --- Utility functions ---
def softmax(x, axis=1):
    """
    Compute softmax values for each set of scores in x.
    
    Args:
        x (np.ndarray): Input array
        axis (int): Axis along which to compute softmax
        
    Returns:
        np.ndarray: Softmax values
    """
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

# --- Strategy A: full multi-head policy (11 heads) ---
def unpack_theta_full(theta: np.ndarray, P: int):
    """
    Unpack theta values for the full multi-head policy (11 heads).
    
    Args:
        theta (np.ndarray): Parameter vector
        P (int): Number of features
    
    Returns:
        dict: Dictionary of parameter heads
    """
    offset = 0
    heads = {}
    # discount head
    heads['disc_w'] = theta[offset:offset+P]
    heads['disc_b'] = theta[offset+P]
    offset += P+1
    
    # PhoneService head
    heads['ps_w'] = theta[offset:offset+P]
    heads['ps_b'] = theta[offset+P]
    offset += P+1
    
    # internet: 3 heads
    heads['int_w'] = theta[offset:offset+3*P].reshape(3, P)
    offset += 3*P
    heads['int_b'] = theta[offset:offset+3]
    offset += 3
    
    # service pairs: 6 heads
    heads['pair_w'] = theta[offset:offset+6*P].reshape(6, P)
    offset += 6*P
    heads['pair_b'] = theta[offset:offset+6]
    
    return heads

def apply_retention_full(
    offered_df: pd.DataFrame,
    theta_full: np.ndarray,
    kmeans
) -> pd.DataFrame:
    """
    Apply the full multi-head retention policy.
    
    Args:
        offered_df (pd.DataFrame): Dataframe of customers offered retention
        theta_full (np.ndarray): Full theta parameter array
        kmeans: K-means model for customer segmentation
    
    Returns:
        pd.DataFrame: Modified dataframe after retention actions
    """
    try:
        df2 = offered_df.copy()
        
        # Ensure all required feature columns exist
        missing_cols = [col for col in feat_cols if col not in df2.columns]
        if missing_cols:
            print(f"Warning: Missing columns in retention input: {missing_cols}")
            for col in missing_cols:
                df2[col] = 0  # Add missing columns with zeros
        
        # Unpack theta and extract features
        heads = unpack_theta_full(theta_full, P)
        X = df2[feat_cols].to_numpy(dtype=np.float32)  # (M, P)
        
        # 1) DISCOUNT
        disc_lin = X.dot(heads['disc_w']) + heads['disc_b']  # (M,)
        discounts = expit(disc_lin)  # (M,)
        df2['MonthlyCharges'] = df2['MonthlyCharges'] * (1.0 - discounts)
        
        # 2) PHONESERVICE: if policy says "on" OR they already had it, keep it on
        ps_lin = X.dot(heads['ps_w']) + heads['ps_b']
        keep_ps = expit(ps_lin) > 0.5  # boolean mask (M,)
        df2['PhoneService'] = (keep_ps | (df2['PhoneService'] == 1)).astype(int)
        
        # Remember who had fiber before we change it
        prev_fiber = (df2['InternetService_Fiber optic'] == 1)
        
        # 3) INTERNET SERVICE (3-way): 0=none,1=DSL,2=Fiber
        int_lin = X.dot(heads['int_w'].T) + heads['int_b'][None, :]  # (M,3)
        int_prob = softmax(int_lin, axis=1)  # (M,3)
        choice = np.argmax(int_prob, axis=1)  # values in {0,1,2}
        
        had_dsl = df2['InternetService_DSL'] == 1
        had_fiber = df2['InternetService_Fiber optic'] == 1
        
        df2['InternetService_DSL'] = (
            (choice == 1) | ((choice != 2) & had_dsl)
        ).astype(int)
        df2['InternetService_Fiber optic'] = (
            (choice == 2) | ((choice != 1) & had_fiber)
        ).astype(int)
        
        # 4) IF you just *lost* Fiber, knock off the one-time charge of 24
        dropped_fiber = prev_fiber & (df2['InternetService_Fiber optic'] == 0)
        df2['MonthlyCharges'] -= 25 * dropped_fiber.astype(int)
        
        # 5) SIX YES/NO PAIRS, but only allow "Yes" if they have any Internet
        has_int = (df2['InternetService_DSL'] == 1) | (df2['InternetService_Fiber optic'] == 1)
        for i, name in enumerate(pair_names):
            lin = X.dot(heads['pair_w'][i]) + heads['pair_b'][i]  # (M,)
            p_yes = expit(lin) > 0.5
            yes = ((p_yes & has_int) | (df2[f"{name}_Yes"] == 1)).astype(int)
            no = (1 - yes).astype(int)
            df2[f"{name}_Yes"] = yes
            df2[f"{name}_No"] = no
        
        # 6) Recompute featurePerCharged
        yes_cols = [c for c in df2.columns if c.endswith("_Yes")]
        if yes_cols:
            df2["featurePerCharged"] = (
                df2[yes_cols].sum(axis=1)
                / np.maximum(df2["MonthlyCharges"], 1e-3)
            )
        else:
            df2["featurePerCharged"] = 0.0
        
        # 7) Re-cluster with kmeans
        cluster_feats = [
            'tenure_binned_1', 'tenure_binned_2', 'tenure_binned_3',
            'Contract_Month-to-month', 'Contract_Two year',
            'TechSupport_Yes', 'TechSupport_No',
            'OnlineSecurity_No', 'OnlineSecurity_Yes',
            'InternetService_DSL', 'InternetService_Fiber optic'
        ]
        
        # Check if all cluster features exist
        missing_cluster_feats = [f for f in cluster_feats if f not in df2.columns]
        if missing_cluster_feats:
            for feat in missing_cluster_feats:
                df2[feat] = 0
        
        # Predict new clusters if kmeans model is available
        if kmeans is not None:
            df2["cluster"] = kmeans.predict(df2[cluster_feats]).astype(int)
        
        print(f"Applied full multi-head retention policy to {len(df2)} customers")
        return df2
        
    except Exception as e:
        print(f"Error in apply_retention_full: {str(e)}")
        print("Returning original dataframe")
        return offered_df

# --- Strategy B: contract-only policy (1 head) ---
def unpack_theta_contract(theta: np.ndarray, P: int):
    """
    Unpack theta values for the contract-only policy (1 head).
    
    Args:
        theta (np.ndarray): Parameter vector
        P (int): Number of features
    
    Returns:
        dict: Dictionary with contract parameters
    """
    heads = {
        'ct_w': theta[:P],    # shape (P,)
        'ct_b': float(theta[P])  # scalar bias
    }
    return heads

def apply_retention_contract(
    offered_df: pd.DataFrame,
    theta_contract: np.ndarray,
    kmeans
) -> pd.DataFrame:
    """
    Apply contract-only retention policy.
    
    Args:
        offered_df (pd.DataFrame): Dataframe of customers offered retention
        theta_contract (np.ndarray): Contract-only theta parameter array
        kmeans: K-means model for customer segmentation
    
    Returns:
        pd.DataFrame: Modified dataframe after retention actions
    """
    try:
        # Unpack theta
        heads = unpack_theta_contract(theta_contract, P)
        df2 = offered_df.copy()
        
        # Ensure all required feature columns exist
        missing_cols = [col for col in feat_cols if col not in df2.columns]
        if missing_cols:
            print(f"Warning: Missing columns in retention input: {missing_cols}")
            for col in missing_cols:
                df2[col] = 0  # Add missing columns with zeros
        
        X = df2[feat_cols].to_numpy(dtype=np.float32)  # (M, P)
        
        # Raw logit → prob of switching to Two-year
        logits = X.dot(heads['ct_w']) + heads['ct_b']  # (M,)
        want_two = expit(logits) > 0.5  # boolean mask
        
        # Eligibility & no-downgrade rules
        eligible = df2['tenure'] >= 4  # min-tenure rule
        still_two = df2['Contract_Two year'] == 1  # once two-year → always two-year
        new_two = (want_two & eligible) | still_two
        
        # Write back
        df2['Contract_Two year'] = new_two.astype(int)
        df2['Contract_Month-to-month'] = (1 - new_two).astype(int)
        
        # Re-cluster on same features used originally
        cluster_feats = [
            'tenure_binned_1', 'tenure_binned_2', 'tenure_binned_3',
            'Contract_Month-to-month', 'Contract_Two year',
            'TechSupport_Yes', 'TechSupport_No',
            'OnlineSecurity_No', 'OnlineSecurity_Yes',
            'InternetService_DSL', 'InternetService_Fiber optic'
        ]
        
        # Check if all cluster features exist
        missing_cluster_feats = [f for f in cluster_feats if f not in df2.columns]
        if missing_cluster_feats:
            for feat in missing_cluster_feats:
                df2[feat] = 0
        
        # Predict new clusters if kmeans model is available
        if kmeans is not None:
            df2['cluster'] = kmeans.predict(df2[cluster_feats]).astype(int)
        
        print(f"Applied contract-only retention policy to {len(df2)} customers")
        return df2
        
    except Exception as e:
        print(f"Error in apply_retention_contract: {str(e)}")
        print("Returning original dataframe")
        return offered_df

def load_thetas():
    """
    Load theta parameters from disk.
    
    Returns:
        tuple: (θ_full, θ_contract) arrays
    """
    try:
        # Attempt to load the theta arrays from disk
        θ_full = np.load("theta_full.npy")
        θ_contract = np.load("theta_contract.npy")
        print(f"Successfully loaded thetas from disk: full={θ_full.shape}, contract={θ_contract.shape}")
        
    except Exception as e:
        print(f"Error loading theta files: {str(e)}")
        print("Creating placeholder theta arrays")
        
        # Create dummy theta arrays with appropriate dimensions
        # Full policy: 11 heads with different dimensions
        # 1 discount head: P+1
        # 1 PhoneService head: P+1
        # 3 internet heads: 3P+3
        # 6 service pair heads: 6P+6
        # Total: P+1 + P+1 + 3P+3 + 6P+6 = 11P+11
        θ_full_size = 11 * P + 11
        
        # Contract policy: just 1 head
        # 1 contract head: P+1
        θ_contract_size = P + 1
        
        # Initialize with small random values
        θ_full = np.random.normal(0, 0.1, size=θ_full_size)
        θ_contract = np.random.normal(0, 0.1, size=θ_contract_size)
        
        print(f"Created placeholder thetas: full={θ_full.shape}, contract={θ_contract.shape}")
    
    return θ_full, θ_contract

