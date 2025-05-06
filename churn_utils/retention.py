
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
def unpack_internet(theta: np.ndarray):
    # skip discount(P+1) + phone(P+1)
    offset = 2*(P+1)
    w = theta[offset      : offset+3*P   ].reshape(3, P)  # (3, P)
    b = theta[offset+3*P  : offset+3*P+3 ]               # (3,)
    return w, b

# ──────────────────────────────────────────────────────────────────────────────
# 2) extract six service‐pair heads (6×P weights + 6 biases) from a full θ
# ──────────────────────────────────────────────────────────────────────────────
def unpack_service_pairs(theta: np.ndarray):
    # skip discount + phone + internet:
    #   2*(P+1) + (3*P + 3) = 5*P + 5  ==  2*(P+1)+3*(P+1) == 5*(P+1)
    offset = 5*(P+1)
    w = theta[offset       : offset+6*P  ].reshape(6, P)  # (6, P)
    b = theta[offset+6*P   : offset+6*P+6]                # (6,)
    return w, b

# ──────────────────────────────────────────────────────────────────────────────
# 3) the “mixed” retention applicator
# ──────────────────────────────────────────────────────────────────────────────
def apply_retention_full(
    offered_df: pd.DataFrame,
    theta_int:  np.ndarray,
    theta_sp:   np.ndarray,
    kmeans
) -> pd.DataFrame:
    df2 = offered_df.copy()
    X   = df2[feat_cols].to_numpy(dtype=np.float32)  # (M, P)

    # —————————————————————————————————————————————————————————————————————————————
    #  A) INTERNET SERVICE head from θ₁
    # —————————————————————————————————————————————————————————————————————————————
    int_w, int_b = unpack_internet(theta_int)
    int_lin      = X.dot(int_w.T) + int_b[None, :]    # (M,3)
    int_prob     = softmax(int_lin, axis=1)           # (M,3)
    choice_int   = np.argmax(int_prob, axis=1)        # 0=none,1=DSL,2=Fiber

    had_dsl      = df2['InternetService_DSL'] == 1
    had_fib      = df2['InternetService_Fiber optic'] == 1

    # keep if already had and you didn't switch away
    df2['InternetService_DSL']         = (
        (choice_int==1) | ((choice_int!=2)&had_dsl)
    ).astype(int)
    df2['InternetService_Fiber optic'] = (
        (choice_int==2) | ((choice_int!=1)&had_fib)
    ).astype(int)

    df2["MonthlyCharges"] -= 25 * (df2['InternetService_DSL'] & had_fib).astype(int)
 
    # —————————————————————————————————————————————————————————————————————————————
    #  B) SERVICE‐PAIRS heads from θ₂
    # —————————————————————————————————————————————————————————————————————————————
    sp_w, sp_b = unpack_service_pairs(theta_sp)
    # only allow “Yes” if they have any Internet
    has_int = (df2['InternetService_DSL']==1) | (df2['InternetService_Fiber optic']==1)

    for i, name in enumerate(pair_names):
        lin  = X.dot(sp_w[i]) + sp_b[i]       # (M,)
        yes  = ((expit(lin)>0.5) & has_int) \
               | (df2[f"{name}_Yes"]==1)     # keep any existing Y=yes
        yes  = yes.astype(int)
        df2[f"{name}_Yes"] = yes
        df2[f"{name}_No"]  = 1 - yes

    # —————————————————————————————————————————————————————————————————————————————
    #  C) recompute featurePerCharged & recluster
    # —————————————————————————————————————————————————————————————————————————————
    yes_cols = [c for c in df2.columns if c.endswith("_Yes")]
    if yes_cols:
        df2["featurePerCharged"] = (
            df2[yes_cols].sum(axis=1)
            / (df2["MonthlyCharges"] + 1e-3)
        )
    else:
        df2["featurePerCharged"] = 0.0

    cluster_feats = [
        'tenure_binned_1','tenure_binned_2','tenure_binned_3',
        'Contract_Month-to-month','Contract_Two year',
        'TechSupport_Yes','TechSupport_No',
        'OnlineSecurity_No','OnlineSecurity_Yes',
        'InternetService_DSL','InternetService_Fiber optic'
    ]
    df2["cluster"] = kmeans.predict(df2[cluster_feats]).astype(int)

    return df2

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
        θ1_full = np.load("theta_full.npy")
        θ2_full = np.load("theta_rename.npy")
        θ_contract = np.load("theta_contract.npy")
        print(f"Successfully loaded thetas from disk: full={θ1_full.shape}, contract={θ_contract.shape}")
        
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
        θ1_full_size = 11 * P + 11
        θ2_full_size = 11 * P + 11
        
        # Contract policy: just 1 head
        # 1 contract head: P+1
        θ_contract_size = P + 1
        
        # Initialize with small random values
        θ1_full = np.random.normal(0, 0.1, size=θ1_full_size)
        θ2_full = np.random.normal(0, 0.1, size=θ1_full_size)
        θ_contract = np.random.normal(0, 0.1, size=θ_contract_size)
        
        print(f"Created placeholder thetas: full={θ1_full.shape}, contract={θ_contract.shape}")
    
    return θ1_full, θ2_full, θ_contract

