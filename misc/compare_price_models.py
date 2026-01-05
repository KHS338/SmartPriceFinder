#!/usr/bin/env python3
"""
Compare All 3 Price Prediction Models

Compares Random Forest, XGBoost, and Neural Network models on:
- Performance (R², MAE, RMSE, Accuracy)
- Training time and efficiency
- Prediction quality
- Feature importance (where applicable)

Generates visualizations and comparison report.

Usage: python compare_price_models.py
"""

import os
import pickle
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(SCRIPT_DIR, 'fullproductscsvs')
MODEL_DIR = os.path.join(SCRIPT_DIR, 'models')
UNIFIED_FILE = os.path.join(CSV_DIR, 'unified_products_with_price_comparison.csv')


def load_data_and_prepare():
    """Load and prepare data for all models."""
    print("\n" + "="*60)
    print("Loading and Preparing Data")
    print("="*60)
    
    df = pd.read_csv(UNIFIED_FILE)
    print(f"[OK] Loaded {len(df)} products")
    
    # Filter valid prices
    df['price_numeric'] = pd.to_numeric(df['price_numeric'], errors='coerce')
    df = df[pd.notna(df['price_numeric']) & (df['price_numeric'] > 0)].copy()
    print(f"[OK] {len(df)} products with valid prices")
    
    # Numeric features
    numeric_features = [
        'size_value', 'normalized_quantity', 'pack_count', 'total_volume',
        'avg_price', 'median_price', 'price_range', 'price_range_pct',
        'num_stores', 'price_percentile', 'store_rank',
        'avg_price_vs_market', 'cheapest_rate'
    ]
    
    for col in numeric_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Binary features
    binary_features = ['is_cheapest', 'is_most_expensive']
    for col in binary_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # Encode categorical
    label_encoders = {}
    categorical_cols = ['site', 'category', 'size_unit', 'brand', 'brand_category']
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str).fillna('unknown'))
            label_encoders[col] = le
    
    # Build feature list
    feature_cols = []
    for col in numeric_features:
        if col in df.columns:
            feature_cols.append(col)
    for col in binary_features:
        if col in df.columns:
            feature_cols.append(col)
    for col in categorical_cols:
        encoded_col = f'{col}_encoded'
        if encoded_col in df.columns:
            feature_cols.append(encoded_col)
    
    X = df[feature_cols].values
    y = df['price_numeric'].values
    
    print(f"[OK] Feature matrix: {X.shape}")
    
    return X, y, feature_cols


def evaluate_model(model, X_test, y_test, model_name, is_neural=False, scaler=None):
    """Evaluate a model and return metrics."""
    if is_neural and scaler is not None:
        X_test = scaler.transform(X_test)
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # Accuracy within tolerance
    def acc_within(tolerance_pct):
        errors = np.abs(y_test - y_pred)
        tolerance = y_test * (tolerance_pct / 100)
        return np.mean(errors <= tolerance) * 100
    
    return {
        'model': model_name,
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'acc_5': acc_within(5),
        'acc_10': acc_within(10),
        'acc_15': acc_within(15),
        'acc_20': acc_within(20)
    }, y_pred


def load_pretrained_models():
    """Load all pre-trained models."""
    print("\n" + "="*60)
    print("Loading Pre-trained Models")
    print("="*60)
    
    models = {}
    
    # Random Forest
    rf_path = os.path.join(MODEL_DIR, 'unified_price_prediction_model.pkl')
    if os.path.exists(rf_path):
        with open(rf_path, 'rb') as f:
            models['Random Forest'] = pickle.load(f)
        print("[OK] Random Forest model loaded")
    else:
        print("[X] Random Forest model not found")
    
    # XGBoost
    xgb_path = os.path.join(MODEL_DIR, 'xgboost_price_model.pkl')
    if os.path.exists(xgb_path):
        with open(xgb_path, 'rb') as f:
            models['XGBoost'] = pickle.load(f)
        print("[OK] XGBoost model loaded")
    else:
        print("[X] XGBoost model not found")
    
    # Neural Network
    nn_path = os.path.join(MODEL_DIR, 'neural_price_model.pkl')
    if os.path.exists(nn_path):
        with open(nn_path, 'rb') as f:
            models['Neural Network'] = pickle.load(f)
        print("[OK] Neural Network model loaded")
    else:
        print("[X] Neural Network model not found")
    
    return models


def compare_models():
    """Compare all models."""
    print("\n" + "="*70)
    print("PRICE PREDICTION MODELS COMPARISON")
    print("="*70)
    
    # Load data
    X, y, feature_cols = load_data_and_prepare()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load models
    models = load_pretrained_models()
    
    if not models:
        print("\n[!] No models found. Train models first.")
        return
    
    # Evaluate each model
    print("\n" + "="*60)
    print("Model Evaluation")
    print("="*60)
    
    results = []
    predictions = {}
    
    for model_name, model_pkg in models.items():
        print(f"\n[Evaluating] {model_name}...")
        
        model = model_pkg['model']
        is_neural = model_name == 'Neural Network'
        scaler = model_pkg.get('scaler') if is_neural else None
        
        metrics, y_pred = evaluate_model(model, X_test, y_test, model_name, is_neural, scaler)
        results.append(metrics)
        predictions[model_name] = y_pred
        
        print(f"  R² Score: {metrics['r2']:.4f}")
        print(f"  MAE: {metrics['mae']:.2f} PKR")
        print(f"  RMSE: {metrics['rmse']:.2f} PKR")
        print(f"  Accuracy (±10%): {metrics['acc_10']:.2f}%")
    
    results_df = pd.DataFrame(results)
    
    # Display comparison table
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON TABLE")
    print("="*70)
    print(results_df.to_string(index=False))
    
    # Create visualizations
    create_visualizations(results_df, predictions, y_test)
    
    # Model complexity comparison
    print_complexity_analysis(models)
    
    # Strengths and weaknesses
    print_strengths_weaknesses()
    
    return results_df, predictions, y_test


def create_visualizations(results_df, predictions, y_test):
    """Create comparison visualizations."""
    print("\n[Creating visualizations...]")
    
    # 1. Performance Metrics Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # R² Score
    axes[0, 0].bar(results_df['model'], results_df['r2'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0, 0].set_title('R² Score (Higher is Better)', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_ylim(0, 1)
    for i, v in enumerate(results_df['r2']):
        axes[0, 0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
    
    # MAE
    axes[0, 1].bar(results_df['model'], results_df['mae'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0, 1].set_title('Mean Absolute Error (Lower is Better)', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('MAE (PKR)')
    for i, v in enumerate(results_df['mae']):
        axes[0, 1].text(i, v + 2, f'{v:.2f}', ha='center', fontweight='bold')
    
    # RMSE
    axes[1, 0].bar(results_df['model'], results_df['rmse'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1, 0].set_title('Root Mean Squared Error (Lower is Better)', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('RMSE (PKR)')
    for i, v in enumerate(results_df['rmse']):
        axes[1, 0].text(i, v + 5, f'{v:.2f}', ha='center', fontweight='bold')
    
    # Accuracy at different tolerances
    tolerances = ['acc_5', 'acc_10', 'acc_15', 'acc_20']
    tolerance_labels = ['±5%', '±10%', '±15%', '±20%']
    x = np.arange(len(tolerance_labels))
    width = 0.25
    
    for i, model_name in enumerate(results_df['model']):
        values = [results_df.loc[results_df['model'] == model_name, tol].values[0] for tol in tolerances]
        axes[1, 1].bar(x + i*width, values, width, label=model_name)
    
    axes[1, 1].set_title('Prediction Accuracy at Different Tolerances', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_xlabel('Tolerance')
    axes[1, 1].set_xticks(x + width)
    axes[1, 1].set_xticklabels(tolerance_labels)
    axes[1, 1].legend()
    axes[1, 1].set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: model_performance_comparison.png")
    
    # 2. Prediction vs Actual scatter plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (model_name, y_pred) in enumerate(predictions.items()):
        axes[idx].scatter(y_test, y_pred, alpha=0.5, s=10)
        axes[idx].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[idx].set_xlabel('Actual Price (PKR)', fontsize=12)
        axes[idx].set_ylabel('Predicted Price (PKR)', fontsize=12)
        axes[idx].set_title(f'{model_name}\nR² = {results_df[results_df["model"]==model_name]["r2"].values[0]:.4f}', 
                           fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_vs_actual.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: prediction_vs_actual.png")
    
    # 3. Residual plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (model_name, y_pred) in enumerate(predictions.items()):
        residuals = y_test - y_pred
        axes[idx].scatter(y_pred, residuals, alpha=0.5, s=10)
        axes[idx].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[idx].set_xlabel('Predicted Price (PKR)', fontsize=12)
        axes[idx].set_ylabel('Residuals (PKR)', fontsize=12)
        axes[idx].set_title(f'{model_name} Residuals', fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('residual_plots.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: residual_plots.png")


def print_complexity_analysis(models):
    """Print complexity and efficiency analysis."""
    print("\n" + "="*70)
    print("COMPLEXITY & EFFICIENCY ANALYSIS")
    print("="*70)
    
    print("\n[Random Forest]")
    if 'Random Forest' in models:
        rf = models['Random Forest']['model']
        print(f"  Estimators: {rf.n_estimators}")
        print(f"  Max Depth: {rf.max_depth}")
        print(f"  Complexity: Medium")
        print(f"  Training Speed: Medium-Fast")
        print(f"  Prediction Speed: Fast")
        print(f"  Memory Usage: Medium")
        print(f"  Interpretability: High (feature importance)")
    
    print("\n[XGBoost]")
    if 'XGBoost' in models:
        xgb = models['XGBoost']['model']
        print(f"  Estimators: {xgb.n_estimators}")
        print(f"  Max Depth: {xgb.max_depth}")
        print(f"  Complexity: Medium-High")
        print(f"  Training Speed: Fast (GPU support)")
        print(f"  Prediction Speed: Very Fast")
        print(f"  Memory Usage: Low-Medium")
        print(f"  Interpretability: High (feature importance)")
    
    print("\n[Neural Network]")
    if 'Neural Network' in models:
        nn = models['Neural Network']['model']
        print(f"  Architecture: {nn.hidden_layer_sizes}")
        print(f"  Activation: {nn.activation}")
        print(f"  Solver: {nn.solver}")
        print(f"  Complexity: High")
        print(f"  Training Speed: Slow (many iterations)")
        print(f"  Prediction Speed: Fast (after scaling)")
        print(f"  Memory Usage: Medium")
        print(f"  Interpretability: Low (black box)")


def print_strengths_weaknesses():
    """Print strengths and weaknesses of each model."""
    print("\n" + "="*70)
    print("STRENGTHS & WEAKNESSES")
    print("="*70)
    
    print("\n[Random Forest]")
    print("  STRENGTHS:")
    print("    + Good baseline performance")
    print("    + Handles non-linear relationships well")
    print("    + Feature importance available")
    print("    + Robust to outliers")
    print("    + No feature scaling needed")
    print("  WEAKNESSES:")
    print("    - Can overfit on noisy data")
    print("    - Large model size")
    print("    - Not best for very large datasets")
    
    print("\n[XGBoost]")
    print("  STRENGTHS:")
    print("    + Usually best accuracy")
    print("    + Fast training with regularization")
    print("    + Handles missing values well")
    print("    + Built-in cross-validation")
    print("    + GPU acceleration available")
    print("  WEAKNESSES:")
    print("    - Requires careful hyperparameter tuning")
    print("    - Can overfit if not regularized")
    print("    - More complex to interpret")
    
    print("\n[Neural Network]")
    print("  STRENGTHS:")
    print("    + Can learn complex patterns")
    print("    + Good for very large datasets")
    print("    + Adaptive learning")
    print("    + Scalable architecture")
    print("  WEAKNESSES:")
    print("    - Requires feature scaling")
    print("    - Long training time")
    print("    - Needs large amounts of data")
    print("    - Difficult to interpret (black box)")
    print("    - Sensitive to initialization")


def main():
    """Main comparison pipeline."""
    print("\n" + "="*70)
    print("PRICE PREDICTION MODELS - COMPREHENSIVE COMPARISON")
    print("="*70)
    
    results_df, predictions, y_test = compare_models()
    
    # Winner determination
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    
    best_r2 = results_df.loc[results_df['r2'].idxmax()]
    best_mae = results_df.loc[results_df['mae'].idxmin()]
    best_acc = results_df.loc[results_df['acc_10'].idxmax()]
    
    print(f"\nBest R² Score: {best_r2['model']} ({best_r2['r2']:.4f})")
    print(f"Best MAE: {best_mae['model']} ({best_mae['mae']:.2f} PKR)")
    print(f"Best Accuracy (±10%): {best_acc['model']} ({best_acc['acc_10']:.2f}%)")
    
    print("\n[OVERALL WINNER]")
    # Count how many times each model is best
    wins = {}
    for model in results_df['model']:
        wins[model] = 0
    wins[best_r2['model']] += 1
    wins[best_mae['model']] += 1
    wins[best_acc['model']] += 1
    
    winner = max(wins, key=wins.get)
    print(f"  {winner} (most metrics won)")
    
    print("\n" + "="*70)
    print("[OK] Comparison complete! Check generated PNG files.")
    print("="*70)


if __name__ == '__main__':
    main()
