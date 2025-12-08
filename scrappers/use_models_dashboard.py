#!/usr/bin/env python3
"""
Use Product Recommendation and Best Deal Models (No Price Prediction)

Analyzes scraped data using only:
1. Product Recommendation Model
2. Best Deal Identification Model

Usage: python use_models_dashboard.py <scraped_csv>
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
        print(f"\n[Brands Available in Training Data: {len(model.brand_groups)}]")
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


def main():
    if len(sys.argv) < 2:
        print("Usage: python use_models_dashboard.py <scraped_csv>")
        print("Example: python use_models_dashboard.py scrapper_pepsi_qtyfixed.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("ANALYZING PRODUCTS WITH ML MODELS")
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
    
    print(f"\n[Loading Models from {models_dir}...]")
    
    rec_model = load_model(rec_model_path) if rec_model_path.exists() else None
    print(f"  {'[OK]' if rec_model else '[X]'} Recommendation Model")
    
    deal_model = load_model(deal_model_path) if deal_model_path.exists() else None
    print(f"  {'[OK]' if deal_model else '[X]'} Best Deal Model")
    
    # Use each model
    use_recommendation_model(df, rec_model)
    use_deal_model(df, deal_model)
    
    print("\n" + "="*60)
    print("[OK] ANALYSIS COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
