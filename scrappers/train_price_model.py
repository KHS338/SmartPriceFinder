

import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Directories
CSV_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fullproductscsvs')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')


def load_features_data():
    """Load all *_features.csv files and combine them."""
    print(f"[i] Loading feature files from {CSV_DIR}...")
    
    search_pattern = os.path.join(CSV_DIR, "*_features.csv")
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        print(f"[!] No *_features.csv files found in {CSV_DIR}")
        print(f"[!] Run csvfeatures.py first to generate feature files")
        return None
    
    print(f"[i] Found {len(csv_files)} feature files:")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")
    
    # Load and combine all CSVs
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
            print(f"[+] Loaded {len(df)} rows from {os.path.basename(csv_file)}")
        except Exception as e:
            print(f"[!] Error loading {csv_file}: {e}")
            continue
    
    if not dfs:
        print("[!] No data loaded")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\n[i] Total combined rows: {len(combined_df)}")
    
    return combined_df


def prepare_features(df):
    """Prepare features for model training."""
    print("\n[i] Preparing features...")
    
    # Required columns
    required_cols = ['price', 'size_value', 'normalized_quantity', 'pack_count', 'size_unit', 'site', 'category']
    
    # Check which columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"[!] Missing required columns: {missing_cols}")
        return None, None, None, None
    
    # Filter to rows with valid price
    df_valid = df.copy()
    df_valid['price'] = pd.to_numeric(df_valid['price'], errors='coerce')
    df_valid = df_valid[pd.notna(df_valid['price']) & (df_valid['price'] > 0)].copy()
    
    print(f"[i] Rows with valid price: {len(df_valid)}")
    
    if len(df_valid) == 0:
        print("[!] No valid price data found")
        return None, None, None, None
    
    # Convert numeric features
    df_valid['size_value'] = pd.to_numeric(df_valid['size_value'], errors='coerce').fillna(0)
    df_valid['normalized_quantity'] = pd.to_numeric(df_valid['normalized_quantity'], errors='coerce').fillna(0)
    df_valid['pack_count'] = pd.to_numeric(df_valid['pack_count'], errors='coerce').fillna(0)
    
    # Add new numeric features if they exist
    if 'total_volume' in df_valid.columns:
        df_valid['total_volume'] = pd.to_numeric(df_valid['total_volume'], errors='coerce').fillna(0)
        print(f"[i] Found total_volume feature")
    
    # Encode categorical features
    label_encoders = {}
    categorical_cols = ['site', 'category', 'size_unit']
    
    # Add new categorical features if they exist
    if 'brand' in df_valid.columns:
        categorical_cols.append('brand')
        print(f"[i] Found brand feature")
    
    if 'brand_category' in df_valid.columns:
        categorical_cols.append('brand_category')
        print(f"[i] Found brand_category feature")
    
    for col in categorical_cols:
        if col in df_valid.columns:
            le = LabelEncoder()
            df_valid[f'{col}_encoded'] = le.fit_transform(df_valid[col].astype(str).fillna('unknown'))
            label_encoders[col] = le
            print(f"[i] Encoded {col}: {len(le.classes_)} unique values")
    
    # Select feature columns
    feature_cols = [
        'size_value',
        'normalized_quantity',
        'pack_count',
        'site_encoded',
        'category_encoded',
        'size_unit_encoded'
    ]
    
    # Add new feature columns if they exist
    if 'total_volume' in df_valid.columns:
        feature_cols.append('total_volume')
    
    if 'brand_encoded' in df_valid.columns:
        feature_cols.append('brand_encoded')
    
    if 'brand_category_encoded' in df_valid.columns:
        feature_cols.append('brand_category_encoded')
    
    # Verify all feature columns exist
    feature_cols = [col for col in feature_cols if col in df_valid.columns]
    
    X = df_valid[feature_cols].values
    y = df_valid['price'].values
    
    print(f"\n[i] Feature matrix shape: {X.shape}")
    print(f"[i] Target vector shape: {y.shape}")
    print(f"[i] Features used: {feature_cols}")
    
    return X, y, label_encoders, feature_cols


def train_model(X, y):
    """Train Random Forest model with 80-20 split."""
    print("\n[i] Splitting data: 80% train, 20% test...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"[i] Training set: {len(X_train)} samples")
    print(f"[i] Test set: {len(X_test)} samples")
    
    print("\n[i] Training Random Forest Regressor...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    print("[+] Model training complete!")
    
    # Evaluate on test set
    print("\n[i] Evaluating model on test set...")
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n=== Model Performance ===")
    print(f"Mean Absolute Error (MAE): {mae:.2f} PKR")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} PKR")
    print(f"R² Score: {r2:.4f}")
    
    # Feature importance
    print(f"\n=== Feature Importance ===")
    # Note: feature_names would need to be passed or stored
    
    return model, X_test, y_test


def save_model(model, label_encoders, feature_cols):
    """Save model and encoders to disk."""
    # Create models directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    model_path = os.path.join(MODEL_DIR, 'price_prediction_model.pkl')
    
    # Package everything together
    model_package = {
        'model': model,
        'label_encoders': label_encoders,
        'feature_cols': feature_cols,
        'version': '1.0',
        'description': 'Price prediction model trained on grocery product features'
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"\n[+] Model saved to: {model_path}")
    print(f"[i] Model file size: {os.path.getsize(model_path) / 1024:.2f} KB")
    
    return model_path


def load_trained_model(model_path=None):
    """Load a saved model (utility function for later use)."""
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, 'price_prediction_model.pkl')
    
    with open(model_path, 'rb') as f:
        model_package = pickle.load(f)
    
    return model_package


def main():
    """Main training pipeline."""
    print("="*60)
    print("Price Prediction Model Training")
    print("="*60)
    
    # Load data
    df = load_features_data()
    if df is None:
        return
    
    # Prepare features
    X, y, label_encoders, feature_cols = prepare_features(df)
    if X is None:
        return
    
    # Train model
    model, X_test, y_test = train_model(X, y)
    
    # Save model
    model_path = save_model(model, label_encoders, feature_cols)
    
    print("\n" + "="*60)
    print("[✓] Training complete!")
    print(f"[✓] Model ready at: {model_path}")
    print("="*60)
    
    # Example of how to load the model later
    print("\n[i] To use the model later:")
    print("    from train_price_model import load_trained_model")
    print("    model_pkg = load_trained_model()")
    print("    predictions = model_pkg['model'].predict(X_new)")


if __name__ == '__main__':
    main()
