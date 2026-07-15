# 📉 Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to churn, built with Python and deployed as an interactive web app using Streamlit.

---

## 🔍 What This Project Does

Customer churn — when a customer stops using a service — is one of the most costly problems in the telecom industry. This project uses real-world telecom data to build a model that identifies customers at high risk of churning, giving businesses the chance to intervene before it's too late.

You enter a customer's details (contract type, tenure, monthly charges, services subscribed, etc.) into the web app, and it instantly predicts whether that customer is likely to churn — along with the probability.

---

## 🌐 Live Demo

> https://telco-churn-olamiji.streamlit.app

---

## 📊 Dataset

**Telco Customer Churn Dataset** — IBM sample dataset available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

- ~7,043 rows, 21 columns
- Features include customer demographics, account information, services subscribed, and churn label
- Target variable: `Churn` (Yes / No)

---

## 🔑 Key Findings from EDA

The exploratory data analysis revealed a consistent, interpretable churn profile:

- **Contract type** is the strongest predictor — month-to-month customers churn at ~43%, compared to just ~3% for two-year contract customers (roughly 15x higher risk)
- **Tenure** — churned customers average ~18 months tenure vs ~37.5 months for retained customers
- **Monthly charges** — churned customers pay ~$74/month on average vs ~$61/month for retained customers
- **Electronic check** payment method has the highest churn rate (~45%), largely because it correlates with month-to-month contracts
- **Fiber optic** internet service has higher churn, again driven by its correlation with month-to-month contracts and higher pricing
- Customers with **no tech support** churn significantly more than those with tech support

**Core churn profile**: new, month-to-month, high-paying customers with no long-term commitment are the highest churn risk.

---

## 🛠️ Technical Details

### Project Structure
```
churn_project/
│
├── churn_notebook.ipynb      # Full EDA, feature engineering, and modeling
├── app.py                    # Streamlit web app
├── churn_model.pkl           # Saved Logistic Regression model
├── feature_columns.json      # Feature column names for input alignment
└── README.md
```

### Workflow

**1. Data Cleaning**
- Fixed `TotalCharges` column stored as string with blank spaces instead of NaN
- Identified 11 rows with tenure = 0 as the source of missing values (set to 0, not dropped)
- Dropped `customerID` (no predictive value) and `TotalCharges` (high multicollinearity with tenure × MonthlyCharges, correlation = 0.83)

**2. Exploratory Data Analysis**
- Analyzed churn rate across contract type, payment method, internet service, tech support, and tenure
- Visualized tenure and monthly charges distributions by churn status
- Built correlation heatmap to identify multicollinearity among numeric features

**3. Feature Engineering**
- Label encoded binary columns (`gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `Churn`)
- One-hot encoded multi-category columns (`Contract`, `PaymentMethod`, `InternetService`, and all service add-on columns)
- Final feature set: 39 columns after encoding

**4. Handling Class Imbalance**
- Dataset has ~73% non-churned vs ~27% churned customers
- Used `class_weight='balanced'` in Logistic Regression to weight minority class appropriately
- Used `scale_pos_weight` for XGBoost equivalent

**5. Modeling**

Three models were tested and compared:

| Model | Recall (Churn) | Precision (Churn) | F1 (Churn) | ROC-AUC |
|-------|---------------|-------------------|------------|---------|
| Logistic Regression | **0.78** | 0.51 | **0.61** | **0.838** |
| Random Forest | 0.48 | 0.60 | 0.53 | 0.814 |
| XGBoost | 0.65 | 0.52 | 0.57 | 0.815 |

**Logistic Regression was selected** as the final model — it outperformed more complex alternatives on recall and ROC-AUC, suggesting the churn patterns in this dataset are largely linear, which aligns with the consistent, interpretable patterns found during EDA.

> Note: Accuracy was deliberately deprioritized as an evaluation metric due to class imbalance. A model predicting "No churn" for every customer would achieve ~73% accuracy while being completely useless. Recall and ROC-AUC are the meaningful metrics here.

**6. Deployment**
- Model saved using `joblib`
- Feature column names saved as `feature_columns.json` to ensure correct input alignment at inference time
- Streamlit app takes raw user inputs, encodes them to match training format using `reindex()`, and returns churn prediction with probability

### Tech Stack

- **Python 3.13**
- **pandas** — data manipulation and EDA
- **matplotlib / seaborn** — data visualization
- **scikit-learn** — modeling, evaluation, train/test split
- **XGBoost** — gradient boosting model
- **joblib** — model serialization
- **Streamlit** — web app deployment

---

## 🚀 How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/Mijipabs/customer-churn-prediction.git
cd customer-churn-prediction
```

**2. Install dependencies**
```bash
pip install pandas scikit-learn xgboost streamlit seaborn matplotlib joblib
```

**3. Run the Streamlit app**
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## 📈 Model Evaluation

The final Logistic Regression model achieves:

- **ROC-AUC: 0.838** — the model correctly ranks a churning customer above a non-churning customer 83.8% of the time
- **Recall (churn class): 0.78** — catches 78% of customers who are actually about to churn
- **Precision (churn class): 0.51** — of customers flagged as churning, 51% actually do

The tradeoff between precision and recall is intentional: for a churn use case, missing a real churner (false negative) is more costly than a false alarm (false positive), so the model is optimized toward higher recall.

---

## 🔮 Future Improvements

- Hyperparameter tuning with GridSearchCV for Random Forest and XGBoost
- SMOTE oversampling as an alternative to `class_weight='balanced'`
- Feature importance chart within the Streamlit app
- Deploy publicly via Streamlit Cloud

---

## 👤 Author

**Onanuga Olamiji**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/onanuga-olamiji-6ba804233)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/Mijipabs)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
