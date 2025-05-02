
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from churn_utils.data import load_data, process_data
from churn_utils.modeling import load_models_and_scaler
from churn_utils.churn import calc_churn_probability, cond_remaining_hybrid, upd
from churn_utils.retention import (
    apply_retention_full, apply_retention_contract, load_thetas
)

# Page configuration
st.set_page_config(
    page_title="Telecom Churn & Retention Dashboard",
    page_icon="📊",
    layout="wide",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 1.5rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #F1F5F9;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- load models + thetas once at startup ---
@st.cache_resource
def load_resources():
    xgb, lr_model, scaler = load_models_and_scaler()
    # TODO: Replace with your actual loading code
    nn_churn = None  # load your pre-trained NN here
    kmeans = None    # load your pre-trained kmeans here
    θ_full, θ_contract = load_thetas()
    return xgb, lr_model, scaler, nn_churn, kmeans, θ_full, θ_contract

xgb, lr_model, scaler, nn_churn, kmeans, θ_full, θ_contract = load_resources()

# --- Main UI ---
st.markdown('<div class="main-header">Telecom Churn & Retention Dashboard</div>', unsafe_allow_html=True)

# --- sidebar: pick retention strategy ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/cell-tower.png", width=80)
    st.title("Dashboard Controls")
    
    strategy = st.radio("Choose retention policy:",
        ["Full multi-head policy", "Contract only"]
    )
    
    if strategy == "Full multi-head policy":
        apply_retention = apply_retention_full
        θ = θ_full
        st.info("Using the comprehensive multi-head policy with 11 action heads")
    else:
        apply_retention = apply_retention_contract
        θ = θ_contract
        st.info("Using contract-only policy with focus on contract extensions")

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Data Overview", "Churn Prediction", "Retention Impact"])

with tab1:
    st.markdown('<div class="section-header">Data Sample</div>', unsafe_allow_html=True)
    
    # Load data (with spinner to show it's working)
    with st.spinner("Loading and processing data..."):
        df = load_data()
        proc = process_data(df)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(proc.head(10), use_container_width=True)
    
    with col2:
        st.markdown("### Dataset Statistics")
        st.write(f"Total customers: {len(proc):,}")
        st.write(f"Feature count: {proc.shape[1]}")
        
        # Basic pie chart for categorical data (example)
        if 'Contract' in proc.columns:
            st.markdown("#### Contract Types")
            contract_counts = proc['Contract'].value_counts()
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.pie(contract_counts, labels=contract_counts.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)

with tab2:
    st.markdown('<div class="section-header">Churn Prediction Demo</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Run churn prediction on full sample", type="primary"):
            with st.spinner("Calculating churn probabilities..."):
                df_demo = proc.copy()
                calc_churn_probability(df_demo, proc, lr_model, nn_churn)
                
                # Display churn probability distribution
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.histplot(df_demo["churn_prob"], bins=30, kde=True, ax=ax)
                ax.set_title("Churn Probability Distribution")
                ax.set_xlabel("Churn Probability")
                ax.set_ylabel("Count")
                st.pyplot(fig)
                
                # Display top at-risk customers
                st.markdown("### Top At-Risk Customers")
                at_risk = df_demo.sort_values("churn_prob", ascending=False).head(10)
                st.dataframe(at_risk[["customerID", "churn_prob", "MonthlyCharges", "tenure"]], 
                           use_container_width=True)
    
    with col2:
        st.markdown("### Customer Churn Factors")
        st.write("Churn prediction combines multiple factors:")
        st.markdown("""
        - Customer demographics
        - Service subscriptions
        - Payment history
        - Tenure and contract type
        - Customer service interactions
        
        The model uses a hybrid approach combining:
        - Nearest-neighbor analysis
        - Logistic regression
        - Tenure-based adjustments
        """)

with tab3:
    st.markdown('<div class="section-header">Retention Impact Simulation</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        n_sample = st.slider("Sample size", 100, 5000, 500, step=100)
        
        run_simulation = st.button("Run Retention Simulation", type="primary")
        
        st.markdown("### Strategy Details")
        if strategy == "Full multi-head policy":
            st.markdown("""
            **Multi-head policy** employs 11 distinct action heads:
            - Contract upgrades
            - Service additions
            - Price adjustments
            - Personalized offers
            - And more...
            """)
        else:
            st.markdown("""
            **Contract-only policy** focuses solely on:
            - Contract extensions (min 4 months)
            - No price/service downgrades
            - Simplified implementation
            """)
    
    with col2:
        if run_simulation:
            with st.spinner("Running retention simulation..."):
                # Run the simulation
                df0 = proc.sample(n_sample, random_state=42).reset_index(drop=True)
                calc_churn_probability(df0, proc, lr_model, nn_churn)
                
                # Create a copy for visualization
                df_before = df0.copy()
                
                # Apply retention strategies
                mask = df0["churn_pred"] == 1
                df_off = df0.copy()
                
                if mask.any():
                    df_off.loc[mask] = apply_retention(df0.loc[mask], θ, kmeans)
                
                calc_churn_probability(df_off, proc, lr_model, nn_churn)
                
                # Calculate remaining life
                R0 = cond_remaining_hybrid(df0, proc, lr_model, nn_churn, kmeans, max_horizon=18)
                R1 = cond_remaining_hybrid(df_off, proc, lr_model, nn_churn, kmeans, max_horizon=18)
                
                # Calculate revenues
                rev0 = ((1-df0["churn_prob"])*df0["MonthlyCharges"]*R0).sum()
                rev1 = ((1-df_off["churn_prob"])*df_off["MonthlyCharges"]*R1).sum()
                
                # Show comparison charts
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                # Before intervention
                sns.histplot(df0["churn_prob"], bins=20, kde=True, ax=ax1, color='salmon')
                ax1.set_title("Churn Probability Before Intervention")
                ax1.set_xlabel("Churn Probability")
                ax1.set_ylim(0, n_sample/3)
                
                # After intervention
                sns.histplot(df_off["churn_prob"], bins=20, kde=True, ax=ax2, color='skyblue')
                ax2.set_title("Churn Probability After Intervention")
                ax2.set_xlabel("Churn Probability")
                ax2.set_ylim(0, n_sample/3)
                
                st.pyplot(fig)
                
                # Display metrics
                st.markdown("### Revenue Impact")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.metric("Revenue Before", f"${rev0:,.0f}", delta=None)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.metric("Revenue After", f"${rev1:,.0f}", delta=None)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    delta = rev1 - rev0
                    delta_percent = (delta / rev0) * 100 if rev0 > 0 else 0
                    st.metric("Revenue Impact", f"${delta:,.0f}", delta=f"{delta_percent:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Show additional metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    avg_churn_before = df0["churn_prob"].mean()
                    avg_churn_after = df_off["churn_prob"].mean()
                    churn_reduction = ((avg_churn_before - avg_churn_after) / avg_churn_before) * 100
                    
                    st.markdown("### Churn Metrics")
                    st.write(f"Average churn probability before: {avg_churn_before:.2%}")
                    st.write(f"Average churn probability after: {avg_churn_after:.2%}")
                    st.write(f"Churn probability reduction: {churn_reduction:.1f}%")
                
                with col2:
                    avg_life_before = R0.mean()
                    avg_life_after = R1.mean()
                    life_increase = avg_life_after - avg_life_before
                    
                    st.markdown("### Customer Lifetime")
                    st.write(f"Average predicted months before: {avg_life_before:.1f}")
                    st.write(f"Average predicted months after: {avg_life_after:.1f}")
                    st.write(f"Average lifetime increase: {life_increase:.1f} months")
