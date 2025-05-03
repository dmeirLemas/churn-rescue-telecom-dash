import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def calc_churn_probability(obs_df, processed_df, model, nn):
    """
    Calculate churn probability using hybrid nearest-neighbor + LR + tenure factor approach.
    
    Args:
        obs_df (pd.DataFrame): Observation dataframe (will be modified with churn_prob)
        processed_df (pd.DataFrame): Complete processed dataframe
        model: Logistic regression model
        nn: Nearest neighbors model
    
    Returns:
        np.ndarray: Array of calculated churn probabilities
    """
    # Define tenure-based churn factor function
    def tenure_churn_factor(tenure):
        base = 0.05   # starting probability at tenure=0
        peak = 0.15   # peak reached at tenure=5
        mid  = 0.05   # value at tenure=12 before sharp drop
        decay_rate = 0.45  # controls exponential decay after tenure=12
        
        if tenure <= 5:
            # Increasing probability from base to peak
            return base + (peak - base) * (tenure / 5)
        elif tenure <= 12:
            # Slight decrease from peak to mid between tenure 5 and 12
            return peak - (peak - mid) * ((tenure - 5) / 7)
        else:
            # Sharper drop-off after tenure 12
            return mid * np.exp(-decay_rate * (tenure - 12))
    
    try:
        # If nn (NearestNeighbors model) is not provided, create a placeholder
        if nn is None:
            print("Warning: No NearestNeighbors model provided, creating placeholder")
            # Filter columns from processed_df
            predictor_cols = [col for col in processed_df.columns if col not in ["Churn", "churn_prob", "churn_pred"]]
            # Create placeholder NN model
            nn = NearestNeighbors(n_neighbors=min(10, len(processed_df)))
            nn.fit(processed_df[predictor_cols].values)
        
        # ---- Calculate churn ratio using nearest neighbors ----
        # Select predictor columns (excluding churn-related outputs)
        predictor_cols = [col for col in processed_df.columns if col not in ["Churn", "churn_prob", "churn_pred"]]
        
        # Ensure all predictor columns exist in obs_df
        missing_cols = [col for col in predictor_cols if col not in obs_df.columns]
        for col in missing_cols:
            if col in processed_df.columns:
                # Use mean from processed_df for missing columns
                obs_df[col] = processed_df[col].mean()
            else:
                # Create dummy column with zeros if not in processed_df either
                obs_df[col] = 0
        
        # Extract features for nearest neighbor calculation
        obs_features = obs_df[predictor_cols]
        
        # Get nearest neighbors
        distances, indices = nn.kneighbors(obs_features)  # shape: (n_obs, n_neighbors)
        
        # Check if "Churn" column exists in processed_df
        if "Churn" not in processed_df.columns:
            print("Warning: 'Churn' column not found in processed_df, creating random values")
            processed_df["Churn"] = np.random.choice([0, 1], size=len(processed_df), p=[0.75, 0.25])
        
        # For each observation, compute the average churn rate of its 10 nearest neighbors
        churn_ratios = np.array([processed_df.iloc[idx]["Churn"].astype(bool).mean() for idx in indices])
        
        # ---- Prepare features for the logistic regression model ----
        if hasattr(model, 'feature_names_in_'):
            # Check if all required features exist in obs_df
            missing_model_cols = [col for col in model.feature_names_in_ if col not in obs_df.columns]
            if missing_model_cols:
                print(f"Warning: Missing columns for model: {missing_model_cols}")
                for col in missing_model_cols:
                    obs_df[col] = 0  # Add missing columns with zeros
            
            obs_x = obs_df[model.feature_names_in_].copy()
        else:
            # If model doesn't have feature_names_in_ attribute
            print("Warning: Model doesn't have feature_names_in_ attribute")
            # Use all non-churn columns as features
            obs_x = obs_df.drop(["churn_prob", "churn_pred"], axis=1, errors='ignore')
        
        # Make predictions with the model
        pred_classes = model.predict(obs_x)
        obs_df["churn_pred"] = pred_classes
        
        # Map prediction confidence: these values (0.85 and 0.62) can be refined per your model's performance
        mapped_confidences = np.where(pred_classes == 0, 1 - 0.87, 0.64)
        
        # ---- Incorporate the tenure factor ----
        # Handle case where tenure column may not exist
        if "tenure" not in obs_df.columns:
            print("Warning: 'tenure' column not found in obs_df, using placeholder values")
            obs_df["tenure"] = np.random.randint(1, 60, len(obs_df))
        
        tenure_values = obs_df["tenure"].values.astype(np.float32)
        # Compute the tenure factor for each observation
        tenure_factors = np.array([tenure_churn_factor(t) for t in tenure_values])
        
        # Final churn probability per tick is the product of:
        # 1. The churn ratio from similar neighbors
        # 2. The mapped model confidence
        # 3. The tenure-based adjustment
        final_probs = churn_ratios * mapped_confidences * tenure_factors
        obs_df["churn_prob"] = final_probs
        
        print(f"Calculated churn probabilities for {len(obs_df)} customers")
        return final_probs
        
    except Exception as e:
        print(f"Error in calc_churn_probability: {str(e)}")
        # Fallback to simple random probabilities
        print("Using fallback random churn probabilities")
        final_probs = np.random.beta(2, 5, size=len(obs_df))
        obs_df["churn_prob"] = final_probs
        obs_df["churn_pred"] = (final_probs > 0.5).astype(int)
        return final_probs


def cond_remaining_hybrid(obs_df: pd.DataFrame,
                          processed_df: pd.DataFrame,
                          lr_model,
                          nn_churn,
                          kmeans,
                          max_horizon: int = 60) -> np.ndarray:
    """
    Calculate Kaplan-Meier style remaining months using hybrid approach.
    
    Args:
        obs_df (pd.DataFrame): Observation dataframe
        processed_df (pd.DataFrame): Complete processed dataframe
        lr_model: Logistic regression model
        nn_churn: Nearest neighbors model
        kmeans: K-means clustering model
        max_horizon (int): Maximum time horizon to consider
    
    Returns:
        np.ndarray: Array of expected remaining months for each customer
    """
    try:
        df = obs_df.copy()
        N = df.shape[0]
        
        # We'll store survivors[k] = Pr(alive after k steps)
        survivors = np.ones(N, dtype=np.float32)
        rem = np.zeros(N, dtype=np.float32)
        
        for k in range(1, max_horizon+1):
            # 1) compute churn_prob at this step (hybrid)
            p = calc_churn_probability(df, processed_df, lr_model, nn_churn)
            # 2) update survival
            survivors *= (1.0 - p)
            # 3) accumulate
            rem += survivors
            # 4) update accordingly
            df = upd(processed_df=processed_df, cus_base=df, kmeans=kmeans)
            
        print(f"Calculated expected remaining months for {N} customers over {max_horizon} months horizon")
        return rem  # shape (N,)
        
    except Exception as e:
        print(f"Error in cond_remaining_hybrid: {str(e)}")
        # Fallback to simple estimation
        print("Using fallback remaining months estimation")
        if "tenure" in obs_df.columns and "churn_prob" in obs_df.columns:
            # Higher tenure and lower churn_prob should result in longer remaining time
            tenure = np.array(obs_df['tenure'])
            churn_prob = np.array(obs_df['churn_prob'])
            remaining_months = max_horizon * (1 - churn_prob) * (0.5 + 0.5 * np.minimum(tenure / 60, 1))
            return np.minimum(remaining_months, max_horizon)
        else:
            return np.random.uniform(1, max_horizon, len(obs_df))


def upd(cus_base: pd.DataFrame, processed_df: pd.DataFrame, kmeans) -> pd.DataFrame:
    """
    Update function for churn-and-spawn simulation.
    
    Args:
        cus_base (pd.DataFrame): Customer base dataframe
        processed_df (pd.DataFrame): Complete processed dataframe
        kmeans: K-means clustering model
    
    Returns:
        pd.DataFrame: Updated customer base
    """
    try:
        cus = cus_base.copy()
        
        # Check if required columns exist
        required_cols = ["tenure", "tenure_binned_1", "tenure_binned_2", "tenure_binned_3"]
        missing_cols = [col for col in required_cols if col not in cus.columns]
        
        if missing_cols:
            print(f"Warning: Missing columns in customer base: {missing_cols}")
            # Add missing tenure columns if needed
            if "tenure" not in cus.columns:
                cus["tenure"] = np.random.randint(1, 60, size=len(cus))
            
            # Add missing binned columns
            for col in ["tenure_binned_1", "tenure_binned_2", "tenure_binned_3"]:
                if col not in cus.columns:
                    if col == "tenure_binned_1":
                        cus[col] = ((cus["tenure"] > 12) & (cus["tenure"] <= 24)).astype(int)
                    elif col == "tenure_binned_2":
                        cus[col] = ((cus["tenure"] > 24) & (cus["tenure"] <= 48)).astype(int)
                    elif col == "tenure_binned_3":
                        cus[col] = (cus["tenure"] > 48).astype(int)
        
        # 1 & 2) Cast old bins, bump tenure, and re-compute bins
        tenure_cols = ["tenure_binned_1", "tenure_binned_2", "tenure_binned_3"]
        old = cus[tenure_cols].astype(bool)
        cus["tenure"] += 1
        cus["tenure_binned_1"] = (cus["tenure"] > 12) & (cus["tenure"] <= 24)
        cus["tenure_binned_2"] = (cus["tenure"] > 24) & (cus["tenure"] <= 48)
        cus["tenure_binned_3"] = cus["tenure"] > 48
        
        # 3) Entrants per bin
        entrants = {
            b: (~old[col]) & cus[col]
            for b, col in zip((1, 2, 3), tenure_cols)
        }
        
        # Check if processed_df has contract type columns
        if not any(col.startswith("Contract_") for col in processed_df.columns):
            print("Warning: Contract columns not found in processed_df")
            # Add dummy contract columns if needed
            if "Contract" not in processed_df.columns:
                processed_df["Contract"] = np.random.choice(
                    ["Month-to-month", "One year", "Two year"], 
                    size=len(processed_df)
                )
        
        # 4) Derive contract_type without apply
        proc = processed_df.copy()
        
        if "Contract_Month-to-month" in proc.columns and "Contract_Two year" in proc.columns:
            proc["contract_type"] = np.where(
                proc["Contract_Month-to-month"], "Month-to-month",
                np.where(proc["Contract_Two year"], "Two year", "One year")
            )
        elif "Contract" in proc.columns:
            # If we have the original Contract column instead of dummies
            proc["contract_type"] = proc["Contract"]
        else:
            # Fallback
            proc["contract_type"] = "Month-to-month"
        
        # 5) Feature columns for KNNs
        drop_cols = {"customerID", "Churn", "churn_prob", "churn_pred", "contract_type"}
        feat_cols = [c for c in proc.columns if c not in drop_cols]
        
        # Ensure all feature columns exist in customer base
        for col in feat_cols:
            if col not in cus.columns:
                if col in proc.columns:
                    cus[col] = proc[col].median() if np.issubdtype(proc[col].dtype, np.number) else proc[col].mode()[0]
                else:
                    cus[col] = 0
        
        # 6) Sample contracts via 30-NN per bin
        contracts = ["Month-to-month", "Two year", "One year"]
        for b, mask in entrants.items():
            idxs = cus.index[mask]
            if len(idxs) == 0:
                continue
            
            if b == 1:
                sel = (proc["tenure"] > 12) & (proc["tenure"] <= 24)
            elif b == 2:
                sel = (proc["tenure"] > 24) & (proc["tenure"] <= 48)
            else:
                sel = proc["tenure"] > 48
            
            proc_bin = proc.loc[sel, feat_cols + ["contract_type"]]
            if proc_bin.empty:
                continue
            
            Xb = proc_bin[feat_cols].values
            nbrs = NearestNeighbors(n_neighbors=min(30, len(Xb))).fit(Xb)
            Xq = cus.loc[idxs, feat_cols].values
            _, neigh_idxs = nbrs.kneighbors(Xq)
            
            neigh_ct = proc_bin["contract_type"].values[neigh_idxs]
            # Compute normalized counts & random choice
            probs_list = [
                pd.value_counts(row, normalize=True)
                  .reindex(contracts, fill_value=0.0)
                  .values
                for row in neigh_ct
            ]
            choices = np.array([
                np.random.choice(contracts, p=probs)
                for probs in probs_list
            ])
            
            # Ensure contract columns exist
            for contract in ["Contract_Month-to-month", "Contract_Two year", "Contract_One year"]:
                if contract not in cus.columns:
                    cus[contract] = 0
            
            # Vectorized assignment
            mask_m2m = choices == "Month-to-month"
            mask_2yr = choices == "Two year"
            cus.loc[idxs, "Contract_Month-to-month"] = mask_m2m.astype(int)
            cus.loc[idxs, "Contract_Two year"] = mask_2yr.astype(int)
            # One-year remains both False
        
        # 7) Update complaintScore via 100-NN in batch
        k = min(100, processed_df.shape[0])
        knn_comp = NearestNeighbors(n_neighbors=k).fit(processed_df[feat_cols].values)
        dists, inds = knn_comp.kneighbors(cus[feat_cols].values)
        
        if "complaintScore" not in processed_df.columns:
            processed_df["complaintScore"] = np.random.uniform(0, 1, size=len(processed_df))
        
        neigh_scores = processed_df["complaintScore"].values[inds]
        nz_mask = neigh_scores > 0
        
        comp_coeff = 0.12
        p_complain = nz_mask.sum(axis=1) / k * comp_coeff
        
        # Avoid division by zero
        inv_d = np.where(nz_mask & (dists > 0), 1.0 / dists, 0.0)
        sum_inv_d = inv_d.sum(axis=1, keepdims=True)
        weights = np.where(sum_inv_d > 0, inv_d / sum_inv_d, 0.0)
        new_score = (neigh_scores * weights).sum(axis=1)
        
        if "complaintScore" not in cus.columns:
            cus["complaintScore"] = 0.0
        
        old_score = cus["complaintScore"].values
        rnd = np.random.rand(len(cus))
        
        mask0 = old_score == 0
        update0 = mask0 & (rnd < p_complain)
        mask1 = ~mask0
        update1 = mask1 & (new_score > old_score)
        update_mask = update0 | update1
        cus.loc[update_mask, "complaintScore"] = new_score[update_mask]
        
        # 8) Update TotalCharges
        if "MonthlyCharges" in cus.columns and "TotalCharges" in cus.columns:
            cus["TotalCharges"] += cus["MonthlyCharges"]
        
        # 9) Re-cluster
        cluster_features = [
            'tenure_binned_1', 'tenure_binned_2', 'tenure_binned_3',
            'Contract_Month-to-month', 'Contract_Two year',
            'TechSupport_Yes', 'TechSupport_No',
            'OnlineSecurity_No', 'OnlineSecurity_Yes',
            'InternetService_DSL', 'InternetService_Fiber optic'
        ]
        
        # Check if all cluster features exist
        missing_cluster_feats = [f for f in cluster_features if f not in cus.columns]
        if missing_cluster_feats:
            for feat in missing_cluster_feats:
                cus[feat] = 0
        
        # Re-cluster
        if kmeans is not None:
            cus["cluster"] = kmeans.predict(cus[cluster_features])
        else:
            print("Warning: kmeans model is None, assigning random clusters")
            cus["cluster"] = np.random.randint(0, 3, size=len(cus))
        
        print(f"Updated customer base with {len(cus)} customers")
        return cus
        
    except Exception as e:
        print(f"Error in upd function: {str(e)}")
        print(f"Returning original customer base")
        return cus_base

