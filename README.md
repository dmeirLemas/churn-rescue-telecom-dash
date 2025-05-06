# 📄 Project Documentation: Churn Rescue & Retention Optimization

KAGGLE NOTEBOOK: https://www.kaggle.com/code/demirelmas/hacklab-sim

## OR YOU COULD JUST GO TO THE WEBSITE TO VIEW THE DASHBOARD

DASHBOARD URL: https://hacklab6foot6.streamlit.app/

## 📁 Project Structure

```
project_root/
│
├── app.py                     # Main Streamlit dashboard app
├── churn_utils/              # All Python helper modules
│   ├── __init__.py
│   ├── retention.py
│   ├── modeling.py
│   ├── processing.py
│   └── ... (others)
├── theta.npy                 # Multi-head retention weights
├── theta_rename.npy          # Additional head (for services or internet)
├── kmeans.pkl                # Pretrained KMeans clustering model
├── lr_model.pkl              # Trained LogisticRegression model
├── nn_churn.pkl              # Trained NearestNeighbors churn model
├── btUTgX.xlsx               # Main customer data
├── newdataset2.csv           # Complaint dataset
├── requirements.txt
└── README.md                 # ← (this file)
```

---

## 🛠️ Requirements

Create a virtual environment and install dependencies:

```bash
python3 -m venv env
source env/bin/activate  # or .\env\Scripts\activate on Windows

pip install -r requirements.txt
```

### `requirements.txt` contents:

```txt
streamlit
pandas
numpy
scikit-learn
xgboost
seaborn
matplotlib
imblearn
joblib
scipy
nltk
openpyxl
```

> 🔹 *Note:* For sentiment scoring in complaints, `nltk` and `vaderSentiment` (or `nltk.sentiment.vader`) are required. Run:
```bash
python -m nltk.downloader vader_lexicon
```

---

## 🚀 Running the App

From the root directory:

```bash
streamlit run app.py
```

Your dashboard will launch in the browser with multiple pages:
- 🗂️ **Data** — View raw and processed samples, feature distributions
- 🤖 **Model** — View LR model coefficients and performance (precision/recall/F1/AUC)
- 🧪 **Retention Simulator** — Apply and compare strategies
- 📊 **Churn Insights** — Service counts vs. churn, cluster breakdowns
- 📈 **Profit Simulator** — Revenue and cost analysis under different strategies

---

## 📌 Notes on Key Components

### 1. **Retention Models**
- **Multi-head model** (`theta.npy`) and (`theta_rename.npy`) applies discounts, services, and internet changes.
- **Contract-only model** (`theta_contract.npy`) changes contract type based on tenure and eligibility.
- A **merged model** applies both in sequence.

### 2. **Key Retention Rules**
- Contract changes only apply if:
  - Tenure ≥ 4 months
  - No downgrade is allowed
- Services are turned on if Internet is available
- Dropping fiber reduces MonthlyCharges by a one-time 25 unit

### 3. **Data Loading**
The main dataset is `btUTgX.xlsx` and complaints are in `newdataset2.csv`. These are merged by `customerID`.

### 4. **Training and Evaluation**
- `lr_model.pkl` is trained.
- Confusion matrix and classification reports are shown in the dashboard.

---


## ✅ Final Checklist Before Running

- [x] Activate virtual environment  
- [x] Run `streamlit run app.py`  
- [x] Ensure `theta.npy`, `theta_rename.npy`, `theta.npy`, and models are in root  
- [x] Confirm that `kmeans.pkl`, `lr_model.pkl`, `nn_churn.pkl` exist  
- [x] Data files (`btUTgX.xlsx` & `newdataset2.csv`) are in root  

---

