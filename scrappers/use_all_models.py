#!/usr/bin/env python3
"""
Use all 3 pre-trained models to analyze scraped product data

Models:
1. Product Recommendation Model
2. Best Deal Identification Model  
3. Unified Price Prediction Model

Usage: python use_all_models.py <scraped_csv>
Example: python use_all_models.py scrapper_pepsi_qtyfixed.csv
"""

import sys
import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path

# Import model classes so pickle can deserialize them
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from train_product_recommendation_model import ProductRecommendationModel
from train_best_deal_model import BestDealModel


def load_model(model_path):
    """Load a pre-trained model"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"[!] Could not load model from {model_path}: {e}")
        return None


def use_recommendation_model(df, model):
    """Use recommendation model to suggest similar products"""
    print("\n" + "="*60)
    print("PRODUCT RECOMMENDATIONS")
    print("="*60)
    
    if model is None:
        print("[!] Recommendation model not loaded")
        return
    
    # Get sample products and find similar ones
    if hasattr(model, 'products_df') and hasattr(model, 'get_similar_products'):
        print("\n[Top Similar Products Based on Training Data]")
        
        # Show recommendations for first few products in scraped data
        for i in range(min(3, len(df))):
            product = df.iloc[i]
            print(f"\n{i+1}. Query: {product['title']}")
            print(f"   Price: Rs. {product['price']:.2f} | Site: {product['site']}")
            
            # Find in training data - escape special regex characters
            search_term = product['title'][:20].replace('(', r'\(').replace(')', r'\)').replace('[', r'\[').replace(']', r'\]')
            matches = model.products_df[
                model.products_df['title'].str.contains(
                    search_term, case=False, na=False, regex=True
                )
            ]
            
            if len(matches) > 0:
                idx = matches.index[0]
                similar = model.get_similar_products(idx, top_n=3)
                
                print("   Similar products:")
                for j, sim_prod in enumerate(similar[:3], 1):
                    print(f"     {j}. {sim_prod['title'][:50]}")
                    print(f"        {sim_prod['site']} - Rs. {sim_prod['price']:.2f} (similarity: {sim_prod['similarity']:.2f})")
            else:
                print("   [No similar products found in training data]")
    
    # Show brand recommendations
    if hasattr(model, 'brand_groups'):
        print("\n[Brands Available in Training Data: {len(model.brand_groups)}]")
        top_brands = list(model.brand_groups.keys())[:5]
        print(f"  Top brands: {', '.join(top_brands)}")
    
    print("\n[OK] Recommendation analysis complete")


def use_deal_model(df, model):
    """Use best deal model to score products"""
    print("\n" + "="*60)
    print("BEST DEAL IDENTIFICATION")
    print("="*60)
    
    if model is None:
        print("[!] Deal model not loaded")
        return
    
    # Calculate deal scores using model's learned statistics
    df_scored = df.copy()
    
    if hasattr(model, 'calculate_deal_score'):
        print("\n[Calculating deal scores using trained model...]")
        
        # Calculate scores for each product
        scores = []
        for _, row in df_scored.iterrows():
            try:
                score = model.predict_deal_score(row)
                scores.append(score)
            except:
                scores.append(0)
        
        df_scored['deal_score'] = scores
        
        # Show top deals
        top_deals = df_scored.nlargest(min(10, len(df_scored)), 'deal_score')
        
        print(f"\n[Top {len(top_deals)} Best Deals from Scraped Data]")
        for i, (_, product) in enumerate(top_deals.iterrows(), 1):
            print(f"\n{i}. {product['title']}")
            print(f"   Site: {product['site']}")
            print(f"   Price: Rs. {product['price']:.2f}")
            print(f"   Deal Score: {product['deal_score']:.1f}/100")
    
    # Show statistics from trained model
    if hasattr(model, 'price_stats'):
        print(f"\n[Model Statistics from Training Data]")
        print(f"  Average price: Rs. {model.price_stats['mean']:.2f}")
        print(f"  Price range: Rs. {model.price_stats['min']:.2f} - Rs. {model.price_stats['max']:.2f}")
        print(f"  Brands tracked: {len(model.brand_stats)}")
        print(f"  Categories tracked: {len(model.category_stats)}")
    
    print("\n[OK] Deal scoring complete")


def use_price_prediction_model(df, model_package):
    """Use price prediction model to predict prices"""
    print("\n" + "="*60)
    print("PRICE PREDICTIONS")
    print("="*60)
    
    if model_package is None:
        print("[!] Price prediction model not loaded")
        return
    
    model = model_package.get('model')
    label_encoders = model_package.get('label_encoders', {})
    feature_cols = model_package.get('feature_cols', [])
    metrics = model_package.get('metrics', {})
    
    print(f"\n[Model Performance Metrics]")
    print(f"  R² Score: {metrics.get('r2', 0):.4f}")
    print(f"  MAE: {metrics.get('mae', 0):.2f} PKR")
    print(f"  RMSE: {metrics.get('rmse', 0):.2f} PKR")
    print(f"  Accuracy (±10%): {metrics.get('acc_10', 0):.2f}%")
    
    print(f"\n[Predicting Prices for Scraped Products...]")
    
    # Prepare features for prediction
    df_pred = df.copy()
    
    # Add numeric features (with defaults)
    numeric_defaults = {
        'size_value': 0,
        'normalized_quantity': 0,
        'pack_count': 0,
        'total_volume': 0,
        'avg_price': df['price'].mean() if 'price' in df.columns else 0,
        'median_price': df['price'].median() if 'price' in df.columns else 0,
        'price_range': 0,
        'price_range_pct': 0,
        'num_stores': 1,
        'price_percentile': 50,
        'store_rank': 1,
        'avg_price_vs_market': 0,
        'cheapest_rate': 0,
        'is_cheapest': 0,
        'is_most_expensive': 0
    }
    
    for col, default in numeric_defaults.items():
        if col not in df_pred.columns:
            df_pred[col] = default
        df_pred[col] = pd.to_numeric(df_pred[col], errors='coerce').fillna(default)
    
    # Encode categorical features
    categorical_cols = ['site', 'category', 'size_unit', 'brand', 'brand_category']
    for col in categorical_cols:
        # Add column if it doesn't exist
        if col not in df_pred.columns:
            df_pred[col] = 'unknown'
        
        if col in label_encoders:
            le = label_encoders[col]
            df_pred[f'{col}_encoded'] = df_pred[col].astype(str).fillna('unknown').apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )
        else:
            df_pred[f'{col}_encoded'] = 0
    
    # Build feature matrix
    X_pred = []
    for col in feature_cols:
        if col in df_pred.columns:
            X_pred.append(df_pred[col].values)
        else:
            X_pred.append(np.zeros(len(df_pred)))
    
    X_pred = np.array(X_pred).T
    
    # Predict prices
    if model and X_pred.shape[0] > 0:
        predictions = model.predict(X_pred)
        df_pred['predicted_price'] = predictions
        df_pred['price_difference'] = df_pred['price'] - df_pred['predicted_price']
        df_pred['price_diff_pct'] = (df_pred['price_difference'] / df_pred['predicted_price']) * 100
        
        print(f"\n[Price Predictions vs Actual]")
        for i, (_, product) in enumerate(df_pred.head(10).iterrows(), 1):
            actual = product['price']
            predicted = product['predicted_price']
            diff_pct = product['price_diff_pct']
            
            status = "[GOOD DEAL]" if diff_pct < -5 else "[OVERPRICED]" if diff_pct > 5 else "[FAIR PRICE]"
            
            print(f"\n{i}. {product['title'][:50]}")
            print(f"   Site: {product['site']}")
            print(f"   Actual Price: Rs. {actual:.2f}")
            print(f"   Predicted Price: Rs. {predicted:.2f}")
            print(f"   Difference: Rs. {diff_pct:+.1f}% {status}")
        
        # Summary
        good_deals = df_pred[df_pred['price_diff_pct'] < -5]
        print(f"\n[Summary]")
        print(f"  Total products analyzed: {len(df_pred)}")
        print(f"  Good deals (>5% below predicted): {len(good_deals)}")
        if len(good_deals) > 0:
            print(f"  Best deal: {good_deals.nsmallest(1, 'price_diff_pct').iloc[0]['title'][:50]}")
    else:
        print("[!] Could not make predictions")
    
    print("\n[OK] Price prediction complete")


def main():
    if len(sys.argv) < 2:
        print("Usage: python use_all_models.py <scraped_csv>")
        print("Example: python use_all_models.py scrapper_pepsi_qtyfixed.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("USING PRE-TRAINED MODELS TO ANALYZE PRODUCTS")
    print("="*60)
    
    # Load scraped data
    df = pd.read_csv(csv_file)
    print(f"\n[OK] Loaded {len(df)} products from {csv_file}")
    print(f"[OK] Sites: {df['site'].unique().tolist()}")
    
    # Load pre-trained models
    models_dir = Path('../models') if os.path.exists('../models') else Path('models')
    if not os.path.exists(models_dir):
        models_dir = Path('models')
    
    rec_model_path = models_dir / 'product_recommendation_model.pkl'
    deal_model_path = models_dir / 'best_deal_model.pkl'
    price_model_path = models_dir / 'unified_price_prediction_model.pkl'
    
    print(f"\n[Loading Models from {models_dir}...]")
    
    rec_model = load_model(rec_model_path) if rec_model_path.exists() else None
    print(f"  {'[OK]' if rec_model else '[X]'} Recommendation Model")
    
    deal_model = load_model(deal_model_path) if deal_model_path.exists() else None
    print(f"  {'[OK]' if deal_model else '[X]'} Best Deal Model")
    
    price_model = load_model(price_model_path) if price_model_path.exists() else None
    print(f"  {'[OK]' if price_model else '[X]'} Price Prediction Model")
    
    # Use each model
    use_recommendation_model(df, rec_model)
    use_deal_model(df, deal_model)
    use_price_prediction_model(df, price_model)
    
    print("\n" + "="*60)
    print("[OK] ALL MODEL ANALYSES COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
