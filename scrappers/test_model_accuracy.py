

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Directories
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'price_prediction_model.pkl')


def load_model():
    """Load the trained model."""
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Model not found at: {MODEL_PATH}")
        print("[!] Run train_price_model.py first to train the model")
        return None
    
    print(f"[i] Loading model from: {MODEL_PATH}")
    with open(MODEL_PATH, 'rb') as f:
        model_package = pickle.load(f)
    
    print(f"[+] Model loaded successfully")
    print(f"[i] Model version: {model_package.get('version', 'unknown')}")
    print(f"[i] Features: {model_package.get('feature_cols', [])}")
    
    return model_package


def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error."""
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_accuracy_within_tolerance(y_true, y_pred, tolerance_percent=10):
    """
    Calculate percentage of predictions within a tolerance range.
    E.g., tolerance_percent=10 means within ±10% of actual price.
    """
    errors = np.abs(y_true - y_pred)
    tolerance = y_true * (tolerance_percent / 100)
    within_tolerance = errors <= tolerance
    accuracy = np.mean(within_tolerance) * 100
    return accuracy


def test_model_on_data(model_package, X_test, y_test):
    """Test model and display comprehensive metrics."""
    model = model_package['model']
    
    print("\n" + "="*60)
    print("Model Evaluation on Test Set")
    print("="*60)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = calculate_mape(y_test, y_pred)
    
    # Accuracy within tolerance ranges
    acc_5 = calculate_accuracy_within_tolerance(y_test, y_pred, tolerance_percent=5)
    acc_10 = calculate_accuracy_within_tolerance(y_test, y_pred, tolerance_percent=10)
    acc_15 = calculate_accuracy_within_tolerance(y_test, y_pred, tolerance_percent=15)
    acc_20 = calculate_accuracy_within_tolerance(y_test, y_pred, tolerance_percent=20)
    
    print(f"\n=== Regression Metrics ===")
    print(f"R² Score (Coefficient of Determination): {r2:.4f}")
    print(f"  → Explains {r2*100:.2f}% of variance in prices")
    print(f"  → 1.0 = perfect, 0.0 = baseline, <0 = worse than baseline")
    
    print(f"\nMean Absolute Error (MAE): {mae:.2f} PKR")
    print(f"  → Average prediction error: ±{mae:.2f} PKR")
    
    print(f"\nRoot Mean Squared Error (RMSE): {rmse:.2f} PKR")
    print(f"  → Penalizes large errors more than MAE")
    
    print(f"\nMean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"  → Average error as percentage of actual price")
    
    print(f"\n=== Prediction Accuracy (within tolerance) ===")
    print(f"Within ±5%:  {acc_5:.2f}% of predictions")
    print(f"Within ±10%: {acc_10:.2f}% of predictions")
    print(f"Within ±15%: {acc_15:.2f}% of predictions")
    print(f"Within ±20%: {acc_20:.2f}% of predictions")
    
    # Sample predictions
    print(f"\n=== Sample Predictions (first 10) ===")
    comparison = pd.DataFrame({
        'Actual': y_test[:10],
        'Predicted': y_pred[:10],
        'Error': y_test[:10] - y_pred[:10],
        'Error %': ((y_test[:10] - y_pred[:10]) / y_test[:10] * 100)
    })
    print(comparison.to_string(index=False))
    
    # Error distribution
    errors = y_test - y_pred
    print(f"\n=== Error Distribution ===")
    print(f"Min error: {errors.min():.2f} PKR")
    print(f"Max error: {errors.max():.2f} PKR")
    print(f"Mean error: {errors.mean():.2f} PKR")
    print(f"Std error: {errors.std():.2f} PKR")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        print(f"\n=== Feature Importance ===")
        feature_cols = model_package.get('feature_cols', [])
        importances = model.feature_importances_
        
        feature_importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print(feature_importance_df.to_string(index=False))
    
    print("\n" + "="*60)
    
    # Summary interpretation
    print("\n=== Model Quality Summary ===")
    if r2 >= 0.8:
        quality = "Excellent"
    elif r2 >= 0.6:
        quality = "Good"
    elif r2 >= 0.4:
        quality = "Fair"
    else:
        quality = "Needs Improvement"
    
    print(f"Overall Model Quality: {quality}")
    print(f"R² Score: {r2:.4f}")
    print(f"Average Error: ±{mae:.2f} PKR ({mape:.2f}%)")
    print(f"Predictions within ±10%: {acc_10:.2f}%")
    
    return {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'accuracy_10pct': acc_10
    }


def main():
    """Load model and test it."""
    # Load the trained model
    model_package = load_model()
    if model_package is None:
        return
    
    # For testing, we need to re-load the test data
    # Since the model was already trained, we need to get the test set again
    print("\n[i] To test the model, we need to reload the data...")
    print("[i] Re-running the training pipeline to get test set...")
    
    # Import functions from training script
    import sys
    sys.path.append(os.path.dirname(__file__))
    from train_price_model import load_features_data, prepare_features
    
    # Load data
    df = load_features_data()
    if df is None:
        return
    
    # Prepare features
    X, y, _, _ = prepare_features(df)
    if X is None:
        return
    
    # Split the same way (with same random_state=42)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"[i] Test set size: {len(X_test)} samples")
    
    # Test the model
    metrics = test_model_on_data(model_package, X_test, y_test)
    
    print("\n[✓] Model testing complete!")


if __name__ == '__main__':
    main()
