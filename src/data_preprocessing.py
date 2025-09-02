import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

def load_data(filepath):
    """Load dataset from CSV file"""
    return pd.read_csv(filepath)

def explore_data(df):
    """Basic data exploration"""
    print(f"Dataset shape: {df.shape}")
    print("\nClass distribution:")
    print(df['Class'].value_counts())
    print(f"\nFraud percentage: {df['Class'].mean():.4%}")
    print("\nMissing values:")
    print(df.isnull().sum().sum())
    
    # Display basic statistics
    print("\nTransaction Amount Statistics:")
    print(df['Amount'].describe())
    
    return df

def preprocess_data(df):
    """Preprocess the dataset"""
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Scale Amount and Time features
    robust_scaler = RobustScaler()
    X['scaled_Amount'] = robust_scaler.fit_transform(X['Amount'].values.reshape(-1, 1))
    X['scaled_Time'] = robust_scaler.fit_transform(X['Time'].values.reshape(-1, 1))
    
    # Drop original Time and Amount
    X = X.drop(['Time', 'Amount'], axis=1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, robust_scaler

def apply_pca(X_train, X_test, n_components=0.95):
    """Apply PCA for dimensionality reduction"""
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    print(f"Original features: {X_train.shape[1]}")
    print(f"PCA components: {X_train_pca.shape[1]}")
    
    return X_train_pca, X_test_pca, pca