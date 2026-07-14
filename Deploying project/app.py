import streamlit as st
import pandas as pd
import joblib
import json

# Load model and feature columns
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, 'churn_model.pkl'))

with open(os.path.join(BASE_DIR, 'feature_columns.json'), 'r') as f:
    feature_columns = json.load(f)

st.title('Customer Churn Prediction')
st.write('Enter customer details below to predict whether they are likely to churn.')

st.subheader('Customer Details')

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox('Gender', ['Male', 'Female'])
    senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
    partner = st.selectbox('Partner', ['No', 'Yes'])
    tenure = st.slider('Tenure (months)', 0, 72, 12)
    phone_service = st.selectbox('Phone Service', ['No', 'Yes'])
    multiple_lines = st.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])

with col2:
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
    online_backup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
    device_protection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
    tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
    streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])

with col3:
    streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox('Paperless Billing', ['No', 'Yes'])
    payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 
                                                      'Bank transfer (automatic)', 
                                                      'Credit card (automatic)'])
    monthly_charges = st.number_input('Monthly Charges ($)', 0.0, 200.0, 50.0)
    dependents = st.selectbox('Dependents', ['No', 'Yes'])

st.subheader('Prediction')

if st.button('Predict Churn'):
    
    # Build raw input dictionary
    input_data = {
        'gender': 1 if gender == 'Female' else 0,
        'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
        'Partner': 1 if partner == 'Yes' else 0,
        'Dependents': 1 if dependents == 'Yes' else 0,
        'tenure': tenure,
        'PhoneService': 1 if phone_service == 'Yes' else 0,
        'PaperlessBilling': 1 if paperless_billing == 'Yes' else 0,
        'MonthlyCharges': monthly_charges,
        'MultipleLines_No': 1 if multiple_lines == 'No' else 0,
        'MultipleLines_No phone service': 1 if multiple_lines == 'No phone service' else 0,
        'MultipleLines_Yes': 1 if multiple_lines == 'Yes' else 0,
        'InternetService_DSL': 1 if internet_service == 'DSL' else 0,
        'InternetService_Fiber optic': 1 if internet_service == 'Fiber optic' else 0,
        'InternetService_No': 1 if internet_service == 'No' else 0,
        'OnlineSecurity_No': 1 if online_security == 'No' else 0,
        'OnlineSecurity_No internet service': 1 if online_security == 'No internet service' else 0,
        'OnlineSecurity_Yes': 1 if online_security == 'Yes' else 0,
        'OnlineBackup_No': 1 if online_backup == 'No' else 0,
        'OnlineBackup_No internet service': 1 if online_backup == 'No internet service' else 0,
        'OnlineBackup_Yes': 1 if online_backup == 'Yes' else 0,
        'DeviceProtection_No': 1 if device_protection == 'No' else 0,
        'DeviceProtection_No internet service': 1 if device_protection == 'No internet service' else 0,
        'DeviceProtection_Yes': 1 if device_protection == 'Yes' else 0,
        'TechSupport_No': 1 if tech_support == 'No' else 0,
        'TechSupport_No internet service': 1 if tech_support == 'No internet service' else 0,
        'TechSupport_Yes': 1 if tech_support == 'Yes' else 0,
        'StreamingTV_No': 1 if streaming_tv == 'No' else 0,
        'StreamingTV_No internet service': 1 if streaming_tv == 'No internet service' else 0,
        'StreamingTV_Yes': 1 if streaming_tv == 'Yes' else 0,
        'StreamingMovies_No': 1 if streaming_movies == 'No' else 0,
        'StreamingMovies_No internet service': 1 if streaming_movies == 'No internet service' else 0,
        'StreamingMovies_Yes': 1 if streaming_movies == 'Yes' else 0,
        'Contract_Month-to-month': 1 if contract == 'Month-to-month' else 0,
        'Contract_One year': 1 if contract == 'One year' else 0,
        'Contract_Two year': 1 if contract == 'Two year' else 0,
        'PaymentMethod_Bank transfer (automatic)': 1 if payment_method == 'Bank transfer (automatic)' else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment_method == 'Credit card (automatic)' else 0,
        'PaymentMethod_Electronic check': 1 if payment_method == 'Electronic check' else 0,
        'PaymentMethod_Mailed check': 1 if payment_method == 'Mailed check' else 0,
    }

    # Convert to dataframe and align with training columns
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # Display result
    if prediction == 1:
        st.error(f'⚠️ This customer is likely to churn (Probability: {probability:.1%})')
    else:
        st.success(f'✅ This customer is unlikely to churn (Probability: {probability:.1%})')