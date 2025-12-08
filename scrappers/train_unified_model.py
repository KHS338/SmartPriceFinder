

import os
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
UNIFIED_FILE = os.path.join(CSV_DIR, 'unified_products_with_price_comparison.csv')


def load_unified_data():
    """Load the unified product dataset."""
    print(f"[i] Loading unified dataset from {UNIFIED_FILE}...")
    
    if not os.path.exists(UNIFIED_FILE):
        print(f"[!] Unified dataset not found. Run fuzzy_match_products.py first.")
        return None
    
    df = pd.read_csv(UNIFIED_FILE)
    print(f"[+] Loaded {len(df)} products")
    print(f"[i] Columns: {len(df.columns)}")
    
    return df


def prepare_features(df):
    """Prepare features for model training with cross-store features."""
    print("\n[i] Preparing features for training...")
    
    # Filter to rows with valid price
    df_valid = df.copy()
    df_valid['price_numeric'] = pd.to_numeric(df_valid['price_numeric'], errors='coerce')
    df_valid = df_valid[pd.notna(df_valid['price_numeric']) & (df_valid['price_numeric'] > 0)].copy()
    
    print(f"[i] Rows with valid price: {len(df_valid)}")
    
    # Convert numeric features
    numeric_features = [
        'size_value', 'normalized_quantity', 'pack_count', 'total_volume',
        'avg_price', 'median_price', 'price_range', 'price_range_pct',
        'num_stores', 'price_percentile', 'store_rank',
        'avg_price_vs_market', 'cheapest_rate'
    ]
    
    for col in numeric_features:
        if col in df_valid.columns:
            df_valid[col] = pd.to_numeric(df_valid[col], errors='coerce').fillna(0)
    
    # Binary features
    binary_features = ['is_cheapest', 'is_most_expensive']
    for col in binary_features:
        if col in df_valid.columns:
            df_valid[col] = pd.to_numeric(df_valid[col], errors='coerce').fillna(0).astype(int)
    
    # Encode categorical features
    label_encoders = {}
    categorical_cols = ['site', 'category', 'size_unit', 'brand', 'brand_category']
    
    for col in categorical_cols:
        if col in df_valid.columns:
            le = LabelEncoder()
            df_valid[f'{col}_encoded'] = le.fit_transform(df_valid[col].astype(str).fillna('unknown'))
            label_encoders[col] = le
            print(f"[i] Encoded {col}: {len(le.classes_)} unique values")
    
    # Build feature list
    feature_cols = []
    
    # Numeric features
    for col in numeric_features:
        if col in df_valid.columns:
            feature_cols.append(col)
    
    # Binary features
    for col in binary_features:
        if col in df_valid.columns:
            feature_cols.append(col)
    
    # Encoded categorical features
    for col in categorical_cols:
        encoded_col = f'{col}_encoded'
        if encoded_col in df_valid.columns:
            feature_cols.append(encoded_col)
    
    print(f"\n[i] Total features selected: {len(feature_cols)}")
    print(f"[i] Features: {feature_cols}")
    
    # Create feature matrix
    X = df_valid[feature_cols].values
    y = df_valid['price_numeric'].values
    
    print(f"\n[i] Feature matrix shape: {X.shape}")
    print(f"[i] Target vector shape: {y.shape}")
    
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
        n_estimators=150,
        max_depth=25,
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
    
    # Calculate MAPE
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # Accuracy within tolerance
    def accuracy_within_tolerance(y_true, y_pred, tolerance_pct):
        errors = np.abs(y_true - y_pred)
        tolerance = y_true * (tolerance_pct / 100)
        return np.mean(errors <= tolerance) * 100
    
    acc_5 = accuracy_within_tolerance(y_test, y_pred, 5)
    acc_10 = accuracy_within_tolerance(y_test, y_pred, 10)
    acc_15 = accuracy_within_tolerance(y_test, y_pred, 15)
    acc_20 = accuracy_within_tolerance(y_test, y_pred, 20)
    
    print(f"\n{'='*60}")
    print("Model Performance on Test Set")
    print(f"{'='*60}")
    print(f"\n=== Regression Metrics ===")
    print(f"R² Score: {r2:.4f} (explains {r2*100:.2f}% of variance)")
    print(f"Mean Absolute Error (MAE): {mae:.2f} PKR")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} PKR")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    
    print(f"\n=== Prediction Accuracy ===")
    print(f"Within ±5%:  {acc_5:.2f}%")
    print(f"Within ±10%: {acc_10:.2f}%")
    print(f"Within ±15%: {acc_15:.2f}%")
    print(f"Within ±20%: {acc_20:.2f}%")
    
    return model, X_test, y_test, {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'acc_10': acc_10
    }


def show_feature_importance(model, feature_cols):
    """Display feature importance."""
    print(f"\n{'='*60}")
    print("Feature Importance")
    print(f"{'='*60}")
    
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print(feature_importance_df.to_string(index=False))
    
    # Show top 10
    print(f"\n=== Top 10 Most Important Features ===")
    print(feature_importance_df.head(10).to_string(index=False))


def save_model(model, label_encoders, feature_cols, metrics):
    """Save model and encoders to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    model_path = os.path.join(MODEL_DIR, 'unified_price_prediction_model.pkl')
    
    # Package everything together
    model_package = {
        'model': model,
        'label_encoders': label_encoders,
        'feature_cols': feature_cols,
        'metrics': metrics,
        'version': '2.0',
        'description': 'Price prediction model with cross-store comparison features'
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"\n[+] Model saved to: {model_path}")
    print(f"[i] Model file size: {os.path.getsize(model_path) / 1024:.2f} KB")
    
    return model_path


def main():
    """Main training pipeline."""
    print("="*60)
    print("Unified Price Prediction Model Training")
    print("With Cross-Store Comparison Features")
    print("="*60)
    
    # Load data
    df = load_unified_data()
    if df is None:
        return
    
    # Prepare features
    X, y, label_encoders, feature_cols = prepare_features(df)
    if X is None:
        return
    
    # Train model
    model, X_test, y_test, metrics = train_model(X, y)
    
    # Show feature importance
    show_feature_importance(model, feature_cols)
    
    # Save model
    model_path = save_model(model, label_encoders, feature_cols, metrics)
    
    print("\n" + "="*60)
    print("[✓] Training complete!")
    print(f"[✓] Model ready at: {model_path}")
    print(f"[✓] R² Score: {metrics['r2']:.4f}")
    print(f"[✓] MAE: {metrics['mae']:.2f} PKR")
    print(f"[✓] Accuracy (±10%): {metrics['acc_10']:.2f}%")
    print("="*60)


if __name__ == '__main__':
    main()
