import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
from sklearn.metrics import average_precision_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import xgboost as xgb
from src.data_preprocessing import load_data, preprocess_data, apply_pca

def train_supervised_model(X_train, X_test, y_train, y_test):
    """Train supervised models for fraud detection"""
    # Handle class imbalance with SMOTE and undersampling
    resampling = Pipeline([
        ('over', SMOTE(sampling_strategy=0.1, random_state=42)),
        ('under', RandomUnderSampler(sampling_strategy=0.5, random_state=42))
    ])
    
    X_res, y_res = resampling.fit_resample(X_train, y_train)
    
    print(f"Original training set shape: {X_train.shape}")
    print(f"Resampled training set shape: {X_res.shape}")
    print(f"Resampled class distribution: {np.bincount(y_res)}")
    
    # Train XGBoost model
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr',
        use_label_encoder=False,
        random_state=42
    )
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.1, 0.01],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=3,
        scoring='average_precision',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_res, y_res)
    
    # Best model
    best_xgb = grid_search.best_estimator_
    
    # Evaluate on test set
    y_pred = best_xgb.predict(X_test)
    y_proba = best_xgb.predict_proba(X_test)[:, 1]
    
    # Print metrics
    print("\nBest Parameters:", grid_search.best_params_)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"Average Precision Score: {average_precision_score(y_test, y_proba):.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Find optimal threshold
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    print(f"\nOptimal threshold: {optimal_threshold:.4f}")
    
    # SHAP values for model interpretability
    explainer = shap.TreeExplainer(best_xgb)
    shap_values = explainer.shap_values(X_test)
    
    return best_xgb, optimal_threshold, explainer, shap_values

def train_unsupervised_model(X_train, X_test, y_test):
    """Train unsupervised model for fraud detection"""
    # Train Isolation Forest
    iso_forest = IsolationForest(
        contamination=0.001,  # Approximate fraud rate
        random_state=42,
        n_jobs=-1
    )
    
    iso_forest.fit(X_train)
    
    # Predict anomalies
    y_pred = iso_forest.predict(X_test)
    y_pred = np.where(y_pred == 1, 0, 1)  # Convert to binary (1: anomaly/fraud)
    
    # Evaluate
    print("\nIsolation Forest Results:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred):.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    return iso_forest

def train_models():
    """Main function to train all models"""
    # Load and preprocess data
    df = load_data('../data/creditcard.csv')
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Apply PCA
    X_train_pca, X_test_pca, pca = apply_pca(X_train, X_test)
    
    # Train supervised model
    print("Training supervised model (XGBoost)...")
    xgb_model, threshold, explainer, shap_values = train_supervised_model(X_train_pca, X_test_pca, y_train, y_test)
    
    # Train unsupervised model
    print("\nTraining unsupervised model (Isolation Forest)...")
    iso_model = train_unsupervised_model(X_train_pca, X_test_pca, y_test)
    
    # Save models and preprocessing objects
    joblib.dump(xgb_model, '../xgb_model.pkl')
    joblib.dump(iso_model, '../iso_model.pkl')
    joblib.dump(scaler, '../scaler.pkl')
    joblib.dump(pca, '../pca.pkl')
    joblib.dump(threshold, '../threshold.pkl')
    
    print("\nAll models and preprocessing objects saved successfully!")
    
    return xgb_model, iso_model, scaler, pca, threshold

if __name__ == "__main__":
    train_models()
