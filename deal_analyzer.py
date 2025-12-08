#!/usr/bin/env python3
"""
ML-based deal identification: scores products and classifies as Excellent/Good/Fair/Poor

Usage: python deal_analyzer.py <input_qtyfixed.csv>
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


def analyze_deals(df):
    """
    Analyze deals and add ML-based scoring columns
    """
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Convert numeric columns
    numeric_cols = ['price', 'size_value', 'pack_count', 'normalized_quantity', 'unit_price', 'per_unit_price']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing critical data
    required_cols = ['price', 'unit_price', 'normalized_quantity']
    df = df.dropna(subset=required_cols)
    
    if len(df) < 5:
        print("[!] Not enough data for analysis (need at least 5 products)")
        return None
    
    print(f"[i] Analyzing {len(df)} products...")
    
    # Feature Engineering
    # 1. Price percentile (lower is better)
    df['price_percentile'] = df['price'].rank(pct=True) * 100
    
    # 2. Unit price percentile (lower is better)
    df['unit_price_percentile'] = df['unit_price'].rank(pct=True) * 100
    
    # 3. Value score: inverse of unit_price, normalized
    df['value_score'] = 1 / (df['unit_price'] + 0.0001)  # Avoid division by zero
    df['value_score'] = (df['value_score'] - df['value_score'].min()) / (df['value_score'].max() - df['value_score'].min()) * 100
    
    # 4. Pack efficiency: products with packs might offer better value
    df['pack_efficiency'] = df['pack_count'].fillna(0).apply(lambda x: min(x / 10, 1) if x > 0 else 0)
    
    # 5. Size tier: larger sizes often have better unit prices
    df['size_tier'] = pd.qcut(df['normalized_quantity'], q=4, labels=['Small', 'Medium', 'Large', 'XL'], duplicates='drop')
    
    # 6. Store encoding for ML
    le_store = LabelEncoder()
    df['store_encoded'] = le_store.fit_transform(df['site'])
    
    # Store rank based on average unit price (lower avg = better store)
    store_avg_price = df.groupby('site')['unit_price'].mean().sort_values()
    store_rank_map = {store: rank for rank, store in enumerate(store_avg_price.index, 1)}
    df['store_rank'] = df['site'].map(store_rank_map)
    df['store_rank_normalized'] = (df['store_rank'] - df['store_rank'].min()) / (df['store_rank'].max() - df['store_rank'].min() + 0.001)
    
    # Calculate Deal Score (0-100, higher = better)
    # Weighted combination of multiple factors
    weights = {
        'value_score': 0.40,           # 40% - most important
        'price_percentile_inv': 0.25,  # 25% - lower price is better
        'pack_efficiency': 0.15,       # 15% - pack deals are good
        'store_rank_inv': 0.20         # 20% - better stores matter
    }
    
    df['price_percentile_inv'] = 100 - df['price_percentile']  # Invert so low price = high score
    df['store_rank_inv'] = 100 - (df['store_rank_normalized'] * 100)
    
    df['deal_score'] = (
        df['value_score'] * weights['value_score'] +
        df['price_percentile_inv'] * weights['price_percentile_inv'] +
        df['pack_efficiency'] * 100 * weights['pack_efficiency'] +
        df['store_rank_inv'] * weights['store_rank_inv']
    )
    
    # Normalize deal_score to 0-100
    df['deal_score'] = df['deal_score'].clip(0, 100)
    
    # Classify into deal tiers based on score
    def classify_deal(score):
        if score >= 75:
            return 'Excellent'
        elif score >= 60:
            return 'Good'
        elif score >= 40:
            return 'Fair'
        else:
            return 'Poor'
    
    df['deal_tier'] = df['deal_score'].apply(classify_deal)
    
    # Machine Learning: Train RandomForest to predict deal quality
    if len(df) >= 10:
        print("[i] Training ML model for deal prediction...")
        
        # Prepare features for ML
        feature_cols = ['price', 'normalized_quantity', 'unit_price', 'pack_count', 'store_encoded']
        X = df[feature_cols].fillna(0)
        
        # Use deal_score as target (regression)
        y = df['deal_score']
        
        # Train Gradient Boosting Regressor
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
        gb_model.fit(X_scaled, y)
        
        # Predict and add ML-enhanced score
        df['ml_predicted_score'] = gb_model.predict(X_scaled)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': gb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n[i] Feature Importance for Deal Scoring:")
        for _, row in feature_importance.iterrows():
            print(f"    {row['feature']}: {row['importance']:.3f}")
        
        # Clustering: Find product groups with similar characteristics
        if len(df) >= 15:
            print("\n[i] Clustering products into groups...")
            n_clusters = min(4, len(df) // 5)  # Max 4 clusters
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df['product_cluster'] = kmeans.fit_predict(X_scaled)
            
            # Analyze clusters
            cluster_stats = df.groupby('product_cluster').agg({
                'price': 'mean',
                'unit_price': 'mean',
                'deal_score': 'mean',
                'site': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Mixed'
            }).round(2)
            
            print("\n[i] Product Clusters:")
            print(cluster_stats)
    
    # Summary Statistics
    print("\n" + "="*60)
    print("DEAL ANALYSIS SUMMARY")
    print("="*60)
    
    tier_counts = df['deal_tier'].value_counts()
    print("\nDeal Tier Distribution:")
    for tier in ['Excellent', 'Good', 'Fair', 'Poor']:
        count = tier_counts.get(tier, 0)
        pct = (count / len(df) * 100) if len(df) > 0 else 0
        print(f"  {tier:12s}: {count:3d} products ({pct:5.1f}%)")
    
    print("\nTop 5 Best Deals:")
    best_deals = df.nlargest(5, 'deal_score')[['site', 'title', 'price', 'unit_price', 'deal_score', 'deal_tier']]
    for idx, row in best_deals.iterrows():
        print(f"  [{row['deal_tier']}] {row['site']:10s} - Rs.{row['price']:6.0f} | Score: {row['deal_score']:.1f}")
        print(f"      {row['title'][:70]}")
    
    print("\nStore Performance (by avg unit price):")
    store_perf = df.groupby('site').agg({
        'unit_price': 'mean',
        'deal_score': 'mean',
        'price': 'count'
    }).round(3).sort_values('unit_price')
    store_perf.columns = ['Avg Unit Price', 'Avg Deal Score', 'Product Count']
    print(store_perf)
    
    # Select columns to keep in output
    output_cols = ['site', 'title', 'price', 'image_url', 'product_url', 
                   'size_value', 'size_unit', 'pack_count', 'normalized_quantity',
                   'notes', 'unit_price', 'unit_price_unit', 'per_unit_price',
                   'deal_score', 'deal_tier', 'price_percentile', 'value_score', 
                   'store_rank']
    
    # Add ML columns if they exist
    if 'ml_predicted_score' in df.columns:
        output_cols.append('ml_predicted_score')
    if 'product_cluster' in df.columns:
        output_cols.append('product_cluster')
    
    # Keep only existing columns
    output_cols = [col for col in output_cols if col in df.columns]
    
    return df[output_cols]


def main():
    if len(sys.argv) < 2:
        print("Usage: python deal_analyzer.py <input_qtyfixed.csv>")
        sys.exit(1)
    
    infile = sys.argv[1]
    
    if not os.path.exists(infile):
        print(f"[!] File not found: {infile}")
        sys.exit(1)
    
    print(f"[+] Loading data from {infile}...")
    df = pd.read_csv(infile)
    
    print(f"[+] Loaded {len(df)} products from {df['site'].nunique()} stores")
    
    # Analyze deals
    analyzed_df = analyze_deals(df)
    
    if analyzed_df is None:
        print("[!] Analysis failed")
        sys.exit(1)
    
    # Save results
    base, ext = os.path.splitext(infile)
    outfile = f"{base}_analyzed.csv"
    
    if os.path.exists(outfile):
        print(f"\n[i] Overwriting existing file: {outfile}")
    
    analyzed_df.to_csv(outfile, index=False)
    print(f"\n[+] Analysis complete! Saved to {outfile}")
    print(f"[+] Added columns: deal_score, deal_tier, price_percentile, value_score, store_rank")
    
    if 'ml_predicted_score' in analyzed_df.columns:
        print(f"[+] ML model trained and predictions added")


if __name__ == '__main__':
    main()
