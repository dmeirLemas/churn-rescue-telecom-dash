# app.py
import streamlit as st
import pandas as pd
import numpy as np

from churn_utils.data      import load_data, process_data
from churn_utils.modeling  import load_models_and_scaler
from churn_utils.churn     import calc_churn_probability, cond_remaining_hybrid
from churn_utils.retention import (
    load_thetas,
    apply_retention_full,
    apply_retention_contract
)

from churn_utils.churn import upd

# ────────────────────────────
# 0) Load everything once
# ────────────────────────────
st.set_page_config(layout="wide")
xgb_model, lr_model, scaler = load_models_and_scaler()
# your NN & kmeans should also be loaded inside churn_utils.modeling if you prefer
from churn_utils.modeling import nn_churn, kmeans
from churn_utils.retention import toggle_costs
θ1_full, θ2_full, θ_contract = load_thetas()

# ────────────────────────────
# Sidebar: navigation
# ────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "📊 Overview",
    "🗂️ Data",
    "🤖 Model",
    "📈 Churn Prediction",
    "🎯 Retention Simulation"
])

# ────────────────────────────
# 1) Overview
# ────────────────────────────
if page == "📊 Overview":
    st.title("📊 Telecom Churn & Retention — Overview")

    st.markdown("""
    Welcome to the *Telecom Churn & Retention* dashboard. This overview introduces the primary components of our project:

    - *Data Sources & Summary*
    - *Our Churn-Prediction Approach*
    - *Two Retention Strategies*

    Use the sidebar to navigate to detailed sections for each part.
    """)

    st.markdown("---")

    st.subheader("Data Sources & Summary")
    st.markdown("""
    - **Customer Records**: demographics, account details, service usage, billing history  
    - **Complaint Logs**: text transcripts used for sentiment analysis  
    - **Churn Labels**: binary indicator for customer churn within the observation window
    """)

    st.markdown("---")

    st.subheader("Our Churn-Prediction Approach")
    st.markdown("""
    1. **Feature Engineering**: service counts, tenure buckets, payment patterns, complaint sentiment  
    2. **Model Training**: Logistic Regression for decision making on churn-risk customers
    3. **Evaluation**: Precision, recall, F1, AUC
    """)

    st.markdown("---")

    st.subheader("Two Retention Strategies")
    st.markdown("""
    **Strategy A: Internet Service Change and Extra Services**  
    - Provide the customer with extra services and promote the change to the DSL service.

    **Strategy B: Contract Upgrade**  
    - Upgrade the length of the contract with discount with respect to previous contract.
    """)

# ────────────────────────────
# 2) Data
# ────────────────────────────
elif page == "🗂️ Data":
    import seaborn as sns
    import matplotlib.pyplot as plt

    st.title("🗂️ Data Exploration")

    # ── 1) Raw data sample ────────────────────────────────────────────────────
    df_raw = load_data()
    st.subheader("Raw Data Sample")
    st.dataframe(df_raw.head(5))

    # ── 2) Processed data sample ─────────────────────────────────────────────
    df = process_data(df_raw)
    st.subheader("Processed Data Sample")
    st.dataframe(df.head(5))

    # ── 3) Shape & memory ─────────────────────────────────────────────────────
    n_rows, n_cols = df.shape
    mem_mb = df.memory_usage(deep=True).sum() / 1024**2
    st.markdown(f"- **Rows:** {n_rows:,} &nbsp; • &nbsp; **Cols:** {n_cols:,}")
    st.markdown(f"- **Memory usage:** {mem_mb:.2f} MB")

    # ── 4) Missing values (%) ─────────────────────────────────────────────────
    st.subheader("Missing Values (%)")
    miss_pct = df.isna().mean() * 100
    miss_pct = miss_pct[miss_pct > 0].sort_values(ascending=False)
    if miss_pct.empty:
        st.write("No missing values!")
    else:
        st.bar_chart(miss_pct)

    # ── 5) Numeric feature summary ─────────────────────────────────────────────
    st.subheader("Numeric Feature Summary")
    num = df.select_dtypes(include=["int", "float"])
    desc = num.describe().T[["min", "mean", "max"]]

    # split by max > 1 vs ≤ 1
    large_mask = desc["max"] > 1.0
    desc_large = desc[large_mask]
    desc_small = desc[~large_mask]

    if not desc_large.empty:
        st.markdown("**📈 High-range features**")
        st.dataframe(desc_large.round(3))
    if not desc_small.empty:
        st.markdown("**🔍 Categorical features**")
        st.dataframe(desc_small.round(3))

    # ── 6) Mean bar-charts split ────────────────────────────────────────────────
    st.subheader("Feature Means")
    col1, col2 = st.columns(2)
    if not desc_large.empty:
        with col1:
            st.markdown("📈 High-range means")
            st.bar_chart(desc_large["mean"])
    if not desc_small.empty:
        with col2:
            st.markdown("🔍 Categorical Ratios")
            st.bar_chart(desc_small["mean"])

    # ── 7) Correlation heatmap ─────────────────────────────────────────────────
    st.subheader("Numeric Feature Correlations")
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.5, ax=ax)
    st.pyplot(fig)

    # ── 8 Churn by Contract Type ────────────────────────────────────────────────
    st.subheader("Churn by Contract Type")
    monthly = df[df["Contract_Month-to-month"] == 1]
    yearly = df[(df["Contract_Two year"] == 0) & (df["Contract_Month-to-month"] == 1)]
    two_years = df[df["Contract_Two year"] == 1]
    churned = df[df["Churn"] == 1]
    # Set up the figure with 3 subplots side by side
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # First plot: Monthly contract
    axes[0].hist(monthly["Churn"], bins=[-0.5, 0.5, 1.5], edgecolor='black', rwidth=0.8)
    axes[0].set_title("Monthly Contract")
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["Not Churned", "Churned"])
    axes[0].set_xlabel("Churn")
    axes[0].set_ylabel("Number of Users")

    # Second plot: Yearly contract
    axes[1].hist(yearly["Churn"], bins=[-0.5, 0.5, 1.5], edgecolor='black', rwidth=0.8)
    axes[1].set_title("Yearly Contract")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Not Churned", "Churned"])
    axes[1].set_xlabel("Churn")
    axes[1].set_ylabel("Number of Users")

    # Third plot: Two years contract
    axes[2].hist(two_years["Churn"], bins=[-0.5, 0.5, 1.5], edgecolor='black', rwidth=0.8)
    axes[2].set_title("Two Years Contract")
    axes[2].set_xticks([0, 1])
    axes[2].set_xticklabels(["Not Churned", "Churned"])
    axes[2].set_xlabel("Churn")
    axes[2].set_ylabel("Number of Users")

    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()

    # ── 9) Churn for each Payment Method ─────────────────────────────────────────────
    mask1 = df['PaymentMethod_Bank transfer (automatic)'] == 1
    mask2 = df['PaymentMethod_Credit card (automatic)'] == 1
    mask3 = df['PaymentMethod_Electronic check'] == 1
    
    st.subheader("Churn by Payment Method")
    electronic_check = df[mask3]
    mailed_check = df[~mask1 & ~mask2 & ~mask3]
    bank_transfer = df[mask1]
    credit_card = df[mask2]
    # Then, plot 4 histograms side-by-side
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    # Plot for Electronic Check
    axes[0].hist(electronic_check["Churn"], bins=[-0.5, 0.5, 1.5], edgecolor='black', rwidth=0.8)
    axes[0].set_title("Electronic Check")
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["Not Churned", "Churned"])
    axes[0].set_xlabel("Churn")
    axes[0].set_ylabel("Number of Users")
    # Plot for Mailed Check
    axes[1].hist(mailed_check["Churn"], bins=[-0.5, 0.5, 1.5], edgecolor='black', rwidth=0.8)
    axes[1].set_title("Mailed Check")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Not Churned", "Churned"])
    axes[1].set_xlabel("Churn")
    axes[1].set_ylabel("Number of Users")
    # Plot for Bank Transfer
    axes[2].hist(bank_transfer["Churn"], bins=[-0.5, 0.5, 1.5], edgecolor='black', rwidth=0.8)
    axes[2].set_title("Bank Transfer (Automatic)")
    axes[2].set_xticks([0, 1])
    axes[2].set_xticklabels(["Not Churned", "Churned"])
    axes[2].set_xlabel("Churn")
    axes[2].set_ylabel("Number of Users")
    # Plot for Credit Card
    axes[3].hist(credit_card["Churn"], bins=[-0.5, 0.5, 1.5], edgecolor='black', rwidth=0.8)
    axes[3].set_title("Credit Card (Automatic)")
    axes[3].set_xticks([0, 1])
    axes[3].set_xticklabels(["Not Churned", "Churned"])
    axes[3].set_xlabel("Churn")
    axes[3].set_ylabel("Number of Users")

    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()

    # ── 10) Churn by Monthly Charges ─────────────────────────────────────────────
    st.subheader("Churn by Monthly Charges")
    sns.set_context("paper", font_scale=1.1)

    # Create the plot
    plt.figure(figsize=(10, 6))
    ax = sns.kdeplot(
        data=df, x="MonthlyCharges", hue="Churn", fill=True,
        palette=["red", "blue"], common_norm=False, alpha=0.5
    )

    # Customize plot
    ax.legend(["Churn", "Not Churn"], loc='upper right')
    ax.set_ylabel('Density')
    ax.set_xlabel('Monthly Charges')
    ax.set_title('Distribution of Monthly Charges by Churn')

    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

    df["MonthlyChargeGroup"] = pd.cut(df["MonthlyCharges"], bins=[0, 35, 70, 150], labels=["Low", "Medium", "High"])
    # Now group by MonthlyChargeGroup and Churn
    monthly_charge_churn = df.groupby(["MonthlyChargeGroup", "Churn"]).size().unstack()

    # Plot
    monthly_charge_churn.plot(kind="bar", edgecolor="black", figsize=(10,6))

    plt.title("Number of Customers by Monthly Charge Group and Churn Status")
    plt.xlabel("Monthly Charge Group")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=0)
    plt.legend(["Not Churned", "Churned"], title="Churn Status")
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

    # ── 11 Churn rate Dependents, Partner & Senior Citizen ───────────────
    st.subheader("Churn Rate by Dependents, Partner & Senior Citizen")
    no_dependents = df[df["Dependents"] == 0]
    has_dependents = df[df["Dependents"] == 1]
    # Partner
    no_partner = df[df["Partner"] == 0]
    has_partner = df[df["Partner"] == 1]
    # Senior Citizen
    not_senior = df[df["SeniorCitizen"] == 0]
    senior_citizen = df[df["SeniorCitizen"] == 1]
    # Now plot side-by-side histograms
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Plot 1: Dependents vs Churn
    axes[0].hist([no_dependents["Churn"], has_dependents["Churn"]], 
                bins=[-0.5, 0.5, 1.5], label=["No Dependents", "Has Dependents"], 
                edgecolor="black", rwidth=0.8)
    axes[0].set_title("Churn by Dependents")
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["Not Churned", "Churned"])
    axes[0].set_xlabel("Churn")
    axes[0].set_ylabel("Number of Users")
    axes[0].legend()

    # Plot 2: Partner vs Churn
    axes[1].hist([no_partner["Churn"], has_partner["Churn"]], 
                bins=[-0.5, 0.5, 1.5], label=["No Partner", "Has Partner"], 
                edgecolor="black", rwidth=0.8)
    axes[1].set_title("Churn by Partner")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Not Churned", "Churned"])
    axes[1].set_xlabel("Churn")
    axes[1].set_ylabel("Number of Users")
    axes[1].legend()

    # Plot 3: Senior Citizen vs Churn
    axes[2].hist([not_senior["Churn"], senior_citizen["Churn"]], 
                bins=[-0.5, 0.5, 1.5], label=["Not Senior", "Senior Citizen"], 
                edgecolor="black", rwidth=0.8)
    axes[2].set_title("Churn by Senior Citizen")
    axes[2].set_xticks([0, 1])
    axes[2].set_xticklabels(["Not Churned", "Churned"])
    axes[2].set_xlabel("Churn")
    axes[2].set_ylabel("Number of Users")
    axes[2].legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()

    # ── 12 Churn by Tenure ─────────────────────────────────────────────
    st.subheader("Churn by Tenure")
    df["TenureGroup"] = pd.cut(df["tenure"], bins=[0, 12, 48, 72], labels=["0-12 months", "1-4 years", "4-6 years"])

    # Calculate churn rate by tenure group
    churn_rates_tenure = df.groupby("TenureGroup")["Churn"].mean()
    # Now number of churned vs not churned customers per group
    tenure_churn = df.groupby(["TenureGroup", "Churn"]).size().unstack()

    # Plot the churned vs not churned counts
    tenure_churn.plot(kind="bar", edgecolor="black", figsize=(10,6))

    plt.title("Churn by Tenure Group")
    plt.xlabel("Tenure Group")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=0)
    plt.legend(["Not Churned", "Churned"], title="Churn Status")
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

    # ─────────────────────────────────────────────────────────────────────────────
    # 13) Churn by Number of Services
    # ─────────────────────────────────────────────────────────────────────────────
    st.subheader("Churn Rate by Number of Services")

    # the “yes” columns from process_data
    service_cols = [
        "PhoneService",
        "MultipleLines_Yes",
        "OnlineSecurity_Yes",
        "OnlineBackup_Yes",
        "DeviceProtection_Yes",
        "TechSupport_Yes",
        "StreamingTV_Yes",
        "StreamingMovies_Yes",
    ]

    # 1) Make sure we’re working off the processed df
    df = process_data(load_data()).copy()

    # 2) Build a “HasInternet” flag from the two dummies
    df["HasInternetService"] = (
        (df["InternetService_DSL"] == 1) |
        (df["InternetService_Fiber optic"] == 1)
    ).astype(int)

    # 3) Count total services per customer
    df["NumServices"] = df[service_cols].sum(axis=1) + df["HasInternetService"]

    # 4) Compute churn‐rate per bucket
    churn_rate_by_services = df.groupby("NumServices")["Churn"].mean()

    # 5) Plot with streamlit
    st.bar_chart(churn_rate_by_services)


# ────────────────────────────
# 3) Model
# ────────────────────────────
elif page == "🤖 Model":
    import seaborn as sns
    import matplotlib.pyplot as plt

    st.title("Churn-Prediction Model")
    st.markdown("""
    We use:
    - **Nearest-neighbors** for local churn ratio  
    - **Logistic regression** for baseline prediction  
    - **Tenure factor** to adjust by customer age  
    """)

    st.subheader("Model Coefficients")
    coefs = pd.Series(lr_model.coef_[0], index=lr_model.feature_names_in_)
    st.dataframe(
        coefs.sort_values(ascending=False)
             .to_frame("Coefficient")
             .round(3)
    )

    st.subheader("Final Test Set Results")
    st.markdown(f"**Accuracy:** 0.8149550402271651")

    st.markdown("**Classification Report**")
    st.markdown("""
| Class            | Precision | Recall | F1-score | Support |
|------------------|----------:|-------:|---------:|--------:|
| Non-Churn (0)    |      0.85 |   0.92 |     0.88 |    1552 |
| Churn (1)        |      0.70 |   0.54 |     0.61 |     561 |
| **accuracy**     |          |        | **0.81** |    2113 |
| **macro avg**    |      0.77 |   0.73 |     0.74 |    2113 |
| **weighted avg** |      0.81 |   0.81 |     0.81 |    2113 |
""")

    st.subheader("Confusion Matrix")
    # your static matrix
    cm = np.array([[1421, 131],
                   [ 260, 301]])
    labels = ["Non-Churn (0)", "Churn (1)"]
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True, fmt="d",
        cmap="Blues", cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)


elif page == "📈 Churn Prediction":

    st.markdown("---")
    st.subheader("🔍 Predict Churn for a Custom Customer")

    with st.expander("Enter customer features manually"):

        # Base profile
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

        # Billing
        monthly = st.number_input("Monthly Charges", min_value=10.0, max_value=200.0, value=70.0)
        total = st.number_input("Total Charges", min_value=10.0, max_value=15000.0, value=800.0)
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        complaint = st.slider("Complaint Score", 0.0, 10.0, 0.0)

        # Contract & payment
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox("Payment Method", [
            "Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"
        ])

        # Internet + Phone
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        internet = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

        # Optional services
        online_sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_prot = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    if st.button("Predict Churn"):
        # Build dictionary of values
        input_dict = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "PaperlessBilling": paperless,
            "complaintScore": complaint,
            "Contract": contract,
            "PaymentMethod": payment,
            "PhoneService": phone,
            "InternetService": internet,
            "MultipleLines": multiple_lines,
            "OnlineSecurity": online_sec,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_prot,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies
        }

        # Create single-row DF and merge with a default row to ensure all columns are available
        raw_df = load_data().iloc[:1].copy()
        for col in input_dict:
            raw_df[col] = input_dict[col]

        # Pass through your full data processor (handles dummies, cluster, feature engineering)
        processed = process_data(raw_df)

        # Recompute churn prediction
        calc_churn_probability(processed, process_data(load_data()), lr_model, nn_churn)
        prob = float(processed["churn_prob"].iloc[0])
        pred = int(processed["churn_pred"].iloc[0])

        st.success(f"Predicted churn probability for next tick: **{prob:.2%}**")
        st.write(f"Prediction: {'Will Churn' if pred else 'Will Stay'}")


# ──────────────────────────────────────────────────────────────────────────
# Retention Strategy Simulator (always compare all three)
# ──────────────────────────────────────────────────────────────────────────
else:
    np.random.seed(60)
    st.title("Retention Strategy Simulator")
    # 1) load & sample once
    processed_df = process_data(load_data())
    n = st.slider("Population size", 100, 2000, 500)
    df0 = processed_df.sample(n).reset_index(drop=True)

    # 2) build three “apply” functions
    apply_full = lambda df: apply_retention_full(df.copy(), θ1_full, θ2_full, kmeans)
    apply_contract = lambda df: apply_retention_contract(df.copy(), θ_contract, kmeans)
    apply_merged = lambda df: apply_retention_contract(
                                apply_retention_full(df.copy(), θ1_full, θ2_full, kmeans),
                                θ_contract,
                                kmeans)

    # BUTTON 1: the “strategy‐comparison” block
    if st.button("Run Strategy Simulator"):
        calc_churn_probability(df0, processed_df, lr_model, nn_churn)
        R0 = cond_remaining_hybrid(df0, processed_df, lr_model, nn_churn, kmeans, max_horizon=18)
        rev0 = ((1 - df0["churn_prob"]) * df0["MonthlyCharges"] * R0).sum()

        mask = df0["churn_pred"] == 1

        # 4) helper to compute metrics for a single strategy
        def compute_metrics(name, apply_fn):
            sub = df0.copy()
            sub.loc[mask] = apply_fn(sub.loc[mask])

            calc_churn_probability(sub, processed_df, lr_model, nn_churn)
            R1 = cond_remaining_hybrid(sub, processed_df, lr_model, nn_churn, kmeans, max_horizon=18)
            rev1 = ((1 - sub["churn_prob"]) * sub["MonthlyCharges"] * R1).sum()

            tog_cols   = list(toggle_costs.keys())
            cost_s     = pd.Series(toggle_costs, dtype=np.float32)
            before_mat = df0.loc[mask, tog_cols]
            after_mat  = sub.loc[mask, tog_cols]
            cost_before= before_mat.mul(cost_s, axis=1).sum(axis=1)
            cost_after = after_mat .mul(cost_s, axis=1).sum(axis=1)
            tog_loss   = (cost_after - cost_before) * R1[mask]
            total_tog  = tog_loss.sum()

            net_gain        = (rev1 - rev0) - total_tog
            profit_post     = rev1 - total_tog
            budget_pc       = net_gain / mask.sum()
            avg_rem         = R0[mask].mean()
            monthly_budget  = budget_pc / avg_rem
            max_disc_pct    = monthly_budget / df0["MonthlyCharges"].mean() * 100

            return {
                "Strategy":         name,
                "Profit before":    rev0,
                "Profit after":     profit_post,
                "Net gain":         net_gain,
                "Flagged churners": int(mask.sum()),
                "Max discount %":   max_disc_pct,
            }

        # 5) compute all three
        results = [
            compute_metrics("Full multi-head",   apply_full),
            compute_metrics("Contract-only",     apply_contract),
            compute_metrics("Merged full→contr", apply_merged),
        ]

        # 6) display side-by-side
        df_res = pd.DataFrame(results).set_index("Strategy")
        st.subheader("Compare all three strategies on the same sample")
        st.dataframe(
            df_res.style.format({
                "Profit before":    "${:,.0f}",
                "Profit after":     "${:,.0f}",
                "Net gain":         "${:,.0f}",
                "Flagged churners": "{:d}",
                "Max discount %":   "{:.1f}%"
            })
        )


    # BUTTON 2: the “simulate vs no‐simulate” block
    if st.button("Run Retention vs No‐Retention Simulation"):

        # recompute processed_df for isolation
        processed_df = process_data(load_data())
        delta_cost = pd.DataFrame.from_dict({"d": [0 for _ in range(300)]})

        def simulate_tick_real(cus_base):
            global delta_cost
            calc_churn_probability(cus_base, processed_df, lr_model, nn_churn)
            mask = (cus_base["churn_pred"]==1) & (cus_base["retent"]==0)
            df_before = cus_base.loc[mask].copy()

            if not df_before.empty:
                offered = apply_merged(df_before.copy())
                offered["retent"] = 1
                calc_churn_probability(offered, processed_df, lr_model, nn_churn)
                cus_base.loc[mask, offered.columns] = offered

            tog_cols = list(toggle_costs.keys())
            cost_s   = pd.Series(toggle_costs, dtype=np.float32)
            cost_before = df_before[tog_cols].mul(cost_s, axis=1).sum(axis=1)
            cost_after  = cus_base.loc[mask, tog_cols].mul(cost_s, axis=1).sum(axis=1)
            tog_loss    = (cost_after - cost_before)
            delta_cost.loc[mask, "d"] = tog_loss

            tick_rev     = cus_base["MonthlyCharges"].sum() - delta_cost["d"].sum()

            randoms   = np.random.rand(len(cus_base))
            churned   = randoms < cus_base["churn_prob"]
            cus_base  = cus_base.loc[~churned].reset_index(drop=True)
            delta_cost= delta_cost.loc[~churned].reset_index(drop=True)

            cus_base = upd(cus_base, processed_df, kmeans)

            return tick_rev, cus_base

        def simulate_tick_fake(cus_base):
            tick_rev = cus_base["MonthlyCharges"].sum()
            calc_churn_probability(cus_base, processed_df, lr_model, nn_churn)
            randoms   = np.random.rand(len(cus_base))
            churned   = randoms < cus_base["churn_prob"]
            cus_base  = cus_base.loc[~churned].reset_index(drop=True)
            cus_base  = upd(cus_base, processed_df, kmeans)
            return tick_rev, cus_base

        cum_real = []
        cum_no   = []
        rev_real = rev_no = 0

        # initialize
        base_real = process_data(load_data()).sample(300, random_state=31).reset_index(drop=True)
        base_real["retent"] = 0
        base_no   = base_real.copy()

        for _ in range(18):
            r, base_real = simulate_tick_real(base_real)
            rev_real   += r
            cum_real.append(rev_real)

            r1, base_no = simulate_tick_fake(base_no)
            rev_no     += r1
            cum_no.append(rev_no)

        # plot
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(cum_no, label="No Retention")
        ax.plot(cum_real, label="With Retention")
        ax.legend()
        st.pyplot(fig)

