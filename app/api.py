from flask import Flask, request, jsonify
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)

# Load models and preprocessing objects
xgb_model = joblib.load('../xgb_model.pkl')
iso_model = joblib.load('../iso_model.pkl')
scaler = joblib.load('../scaler.pkl')
pca = joblib.load('../pca.pkl')
threshold = joblib.load('../threshold.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for fraud prediction"""
    try:
        # Get JSON data
        data = request.json
        features = data['features']
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Preprocess
        df['scaled_Amount'] = scaler.transform(df['Amount'].values.reshape(-1, 1))
        df['scaled_Time'] = scaler.transform(df['Time'].values.reshape(-1, 1))
        df = df.drop(['Time', 'Amount'], axis=1)
        
        # Apply PCA
        df_pca = pca.transform(df)
        
        # Predict with XGBoost
        xgb_proba = xgb_model.predict_proba(df_pca)[0, 1]
        xgb_pred = 1 if xgb_proba > threshold else 0
        
        # Predict with Isolation Forest
        iso_pred = iso_model.predict(df_pca)[0]
        iso_pred = 0 if iso_pred == 1 else 1  # Convert to binary
        
        # Combine predictions (use XGBoost as primary)
        final_pred = xgb_pred
        confidence = xgb_proba if xgb_pred == 1 else 1 - xgb_proba
        
        # Return response
        response = {
            'fraud_prediction': int(final_pred),
            'confidence': float(confidence),
            'xgb_prediction': int(xgb_pred),
            'xgb_probability': float(xgb_proba),
            'isolation_forest_prediction': int(iso_pred)
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

@app.route('/models/info', methods=['GET'])
def model_info():
    """Get information about the models"""
    return jsonify({
        'models': {
            'xgboost': {
                'type': 'XGBoost Classifier',
                'version': '1.5.1',
                'threshold': float(threshold)
            },
            'isolation_forest': {
                'type': 'Isolation Forest',
                'version': '0.5.0'
            }
        },
        'preprocessing': {
            'scaler': 'StandardScaler',
            'pca': f'PCA (n_components={pca.n_components_})'
        }
    })

@app.route('/data/stats', methods=['GET'])
def data_stats():
    """Get statistics about the training data"""
    try:
        df = pd.read_csv('../data/creditcard.csv')
        stats = {
            'total_transactions': len(df),
            'fraud_percentage': float(df['Class'].mean() * 100),
            'amount_stats': {
                'mean': float(df['Amount'].mean()),
                'std': float(df['Amount'].std()),
                'min': float(df['Amount'].min()),
                'max': float(df['Amount'].max())
            }
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Handle batch predictions"""
    try:
        data = request.json
        transactions = data['transactions']
        
        results = []
        for transaction in transactions:
            df = pd.DataFrame([transaction])
            
            # Preprocess
            df['scaled_Amount'] = scaler.transform(df['Amount'].values.reshape(-1, 1))
            df['scaled_Time'] = scaler.transform(df['Time'].values.reshape(-1, 1))
            df = df.drop(['Time', 'Amount'], axis=1)
            
            # Apply PCA
            df_pca = pca.transform(df)
            
            # Predict
            xgb_proba = xgb_model.predict_proba(df_pca)[0, 1]
            xgb_pred = 1 if xgb_proba > threshold else 0
            iso_pred = iso_model.predict(df_pca)[0]
            iso_pred = 0 if iso_pred == 1 else 1
            
            results.append({
                'fraud_prediction': int(xgb_pred),
                'confidence': float(xgb_proba if xgb_pred == 1 else 1 - xgb_proba),
                'xgb_probability': float(xgb_proba),
                'isolation_forest_prediction': int(iso_pred)
            })
        
        return jsonify({'predictions': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
