import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import requests
import time

# Load models and preprocessing objects
xgb_model = joblib.load('../xgb_model.pkl')
iso_model = joblib.load('../iso_model.pkl')
scaler = joblib.load('../scaler.pkl')
pca = joblib.load('../pca.pkl')
threshold = joblib.load('../threshold.pkl')

# App title
st.title('Credit Card Fraud Detection')

# Sidebar for inputs
st.sidebar.header('Transaction Information')

def user_input_features():
    """Create input widgets for transaction data"""
    # Time in seconds (0 to 172800 = 48 hours)
    time = st.sidebar.slider('Time (seconds)', 0, 172800, 36000)
    
    # Transaction amount
    amount = st.sidebar.number_input('Transaction Amount', min_value=0.0, value=50.0)
    
    # V1-V28 features (PCA transformed features)
    v_features = {}
    for i in range(1, 29):
        v_features[f'V{i}'] = st.sidebar.slider(f'V{i}', -5.0, 5.0, 0.0)
    
    # Create dataframe
    data = {
        'Time': time,
        'Amount': amount
    }
    
    # Add V features
    for i in range(1, 29):
        data[f'V{i}'] = v_features[f'V{i}']
    
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Display input data
st.subheader('Transaction Information')
st.write(input_df)

# Preprocess input
input_df['scaled_Amount'] = scaler.transform(input_df['Amount'].values.reshape(-1, 1))
input_df['scaled_Time'] = scaler.transform(input_df['Time'].values.reshape(-1, 1))
input_df = input_df.drop(['Time', 'Amount'], axis=1)

# Apply PCA
input_pca = pca.transform(input_df)

# Make prediction
xgb_proba = xgb_model.predict_proba(input_pca)[0, 1]
xgb_pred = 1 if xgb_proba > threshold else 0

iso_pred = iso_model.predict(input_pca)[0]
iso_pred = 0 if iso_pred == 1 else 1  # Convert to binary

# Final prediction (using XGBoost as primary)
final_pred = xgb_pred
confidence = xgb_proba if xgb_pred == 1 else 1 - xgb_proba

# Display prediction
st.subheader('Fraud Detection Results')
if final_pred == 1:
    st.error('⚠️ Fraudulent Transaction Detected')
else:
    st.success('✅ Transaction Appears Legitimate')

st.write(f"Confidence: {confidence:.2%}")

# Model comparison
st.subheader('Model Comparison')
col1, col2 = st.columns(2)

with col1:
    st.write("**XGBoost Model**")
    st.write(f"Prediction: {'Fraud' if xgb_pred == 1 else 'Legitimate'}")
    st.write(f"Probability: {xgb_proba:.4f}")

with col2:
    st.write("**Isolation Forest Model**")
    st.write(f"Prediction: {'Fraud' if iso_pred == 1 else 'Legitimate'}")

# Feature importance
st.subheader('Feature Importance')
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(input_pca)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, input_pca, plot_type="bar", max_display=10, show=False)
plt.tight_layout()
st.pyplot(plt)

# Real-time simulation
st.subheader('Real-time Transaction Simulation')
if st.button('Simulate Transaction'):
    # Simulate API call
    with st.spinner('Processing transaction...'):
        # Prepare data for API
        features = input_df.to_dict(orient='records')[0]
        # Add back Time and Amount for API
        features['Time'] = user_input_features()['Time'].values[0]
        features['Amount'] = user_input_features()['Amount'].values[0]
        
        # Call API
        try:
            response = requests.post(
                'http://localhost:5000/predict',
                json={'features': features},
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                st.write(f"Fraud Prediction: {'Yes' if result['fraud_prediction'] == 1 else 'No'}")
                st.write(f"Confidence: {result['confidence']:.2%}")
            else:
                st.error("API Error: " + response.text)
        except Exception as e:
            st.error(f"Error connecting to API: {str(e)}")