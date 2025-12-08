#!/usr/bin/env python3
"""
Fuzzy match products across different stores to identify same products.
Creates a unified product dataset with cross-store price comparisons.

Process:
1. Load all *_features.csv files
2. Normalize product names (remove extra words, standardize units)
3. Fuzzy match products across stores
4. Generate cross-store price features
5. Save unified dataset with price comparison features

Usage: python fuzzy_match_products.py
"""

import os
import glob
import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process
import re

# Directories
CSV_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fullproductscsvs')
OUTPUT_FILE = os.path.join(CSV_DIR, 'unified_products_with_price_comparison.csv')


def normalize_product_name(title, size_value, size_unit):
    """
    Normalize product name for fuzzy matching.
    Removes noise, standardizes format.
    """
    if not isinstance(title, str):
        title = str(title or '')
    
    # Convert to lowercase
    normalized = title.lower()
    
    # Remove common filler words
    filler_words = ['fresh', 'pure', 'best', 'quality', 'original', 'new', 'premium', 
                    'special', 'super', 'extra', 'deluxe', 'imported', 'local']
    for word in filler_words:
        normalized = re.sub(r'\b' + word + r'\b', '', normalized)
    
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    # Add standardized size if available
    if pd.notna(size_value) and size_value != '' and pd.notna(size_unit) and size_unit != '':
        # Standardize units
        unit = str(size_unit).lower()
        if unit in ['kg', 'g']:
            # Convert to grams for consistency
            if unit == 'kg':
                size_in_g = float(size_value) * 1000
            else:
                size_in_g = float(size_value)
            normalized += f" {int(size_in_g)}g"
        elif unit in ['l', 'ml']:
            # Convert to ml for consistency
            if unit == 'l':
                size_in_ml = float(size_value) * 1000
            else:
                size_in_ml = float(size_value)
            normalized += f" {int(size_in_ml)}ml"
        else:
            normalized += f" {size_value}{unit}"
    
    return normalized


def load_all_products():
    """Load all feature CSV files and combine them."""
    print(f"[i] Loading products from {CSV_DIR}...")
    
    search_pattern = os.path.join(CSV_DIR, "*_features.csv")
    csv_files = glob.glob(search_pattern)
    
    # Filter out the unified output file to avoid loading it as input
    csv_files = [f for f in csv_files if 'unified_products_with_price_comparison' not in f]
    
    if not csv_files:
        print(f"[!] No *_features.csv files found")
        return None
    
    print(f"[i] Found {len(csv_files)} feature files")
    
    all_products = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Required columns
            if 'title' not in df.columns or 'price' not in df.columns or 'site' not in df.columns:
                print(f"[!] Skipping {os.path.basename(csv_file)}: missing required columns")
                continue
            
            # Add source file info
            df['source_file'] = os.path.basename(csv_file)
            
            all_products.append(df)
            print(f"[+] Loaded {len(df)} products from {os.path.basename(csv_file)}")
            
        except Exception as e:
            print(f"[!] Error loading {csv_file}: {e}")
            continue
    
    if not all_products:
        return None
    
    combined = pd.concat(all_products, ignore_index=True)
    print(f"\n[i] Total products loaded: {len(combined)}")
    
    return combined


def create_normalized_names(df):
    """Add normalized product name column for matching."""
    print("\n[i] Creating normalized product names for matching...")
    
    df['normalized_name'] = df.apply(
        lambda row: normalize_product_name(
            row.get('title', ''),
            row.get('size_value', ''),
            row.get('size_unit', '')
        ),
        axis=1
    )
    
    print(f"[+] Normalized {len(df)} product names")
    return df


def fuzzy_match_products(df, similarity_threshold=85):
    """
    Group products across stores using fuzzy matching.
    Returns dataframe with product_group_id assigned.
    """
    print(f"\n[i] Starting fuzzy matching (threshold: {similarity_threshold}%)...")
    
    # Filter valid products with prices
    df['price_numeric'] = pd.to_numeric(df['price'], errors='coerce')
    valid_df = df[pd.notna(df['price_numeric']) & (df['price_numeric'] > 0)].copy()
    
    print(f"[i] Matching {len(valid_df)} products with valid prices")
    
    # Get unique normalized names
    unique_names = valid_df['normalized_name'].unique()
    print(f"[i] Found {len(unique_names)} unique normalized names")
    
    # Group similar products
    product_groups = []
    processed = set()
    group_id = 0
    
    print("[i] Grouping similar products...")
    for idx, name in enumerate(unique_names):
        if name in processed:
            continue
        
        if idx % 500 == 0:
            print(f"[i] Processed {idx}/{len(unique_names)} names, created {group_id} groups")
        
        # Find similar names
        matches = process.extract(
            name,
            unique_names,
            scorer=fuzz.token_sort_ratio,
            limit=10
        )
        
        # Group names above threshold
        group = [name]
        for match_name, score, _ in matches:
            if match_name != name and score >= similarity_threshold and match_name not in processed:
                group.append(match_name)
                processed.add(match_name)
        
        processed.add(name)
        
        # Only create group if we have products from multiple stores OR high confidence match
        products_in_group = valid_df[valid_df['normalized_name'].isin(group)]
        unique_stores = products_in_group['site'].nunique()
        
        if unique_stores >= 2 or len(group) > 1:  # Multiple stores or high similarity
            for group_name in group:
                product_groups.append({
                    'normalized_name': group_name,
                    'product_group_id': group_id
                })
            group_id += 1
    
    print(f"\n[+] Created {group_id} product groups")
    
    # Create mapping dataframe
    groups_df = pd.DataFrame(product_groups)
    
    # Merge back to original
    valid_df = valid_df.merge(groups_df, on='normalized_name', how='left')
    
    # Assign unique IDs to ungrouped products
    ungrouped_mask = valid_df['product_group_id'].isna()
    ungrouped_count = ungrouped_mask.sum()
    
    if ungrouped_count > 0:
        valid_df.loc[ungrouped_mask, 'product_group_id'] = range(group_id, group_id + ungrouped_count)
        print(f"[i] Assigned unique IDs to {ungrouped_count} ungrouped products")
    
    valid_df['product_group_id'] = valid_df['product_group_id'].astype(int)
    
    return valid_df


def generate_cross_store_features(df):
    """
    Generate price comparison features across stores for each product group.
    """
    print("\n[i] Generating cross-store price comparison features...")
    
    # Group by product_group_id
    grouped = df.groupby('product_group_id')
    
    # Calculate aggregate statistics
    print("[i] Calculating price statistics per product group...")
    
    price_stats = grouped['price_numeric'].agg([
        ('min_price', 'min'),
        ('max_price', 'max'),
        ('avg_price', 'mean'),
        ('median_price', 'median'),
        ('std_price', 'std'),
        ('num_stores', 'count')
    ]).reset_index()
    
    # Fill NaN std with 0 (single store products)
    price_stats['std_price'] = price_stats['std_price'].fillna(0)
    
    # Calculate price range
    price_stats['price_range'] = price_stats['max_price'] - price_stats['min_price']
    price_stats['price_range_pct'] = (price_stats['price_range'] / price_stats['avg_price'] * 100).fillna(0)
    
    # Merge back to main dataframe
    df = df.merge(price_stats, on='product_group_id', how='left')
    
    # Calculate per-row features
    print("[i] Calculating per-product price features...")
    
    # Price vs average
    df['price_vs_avg'] = df['price_numeric'] - df['avg_price']
    df['price_vs_avg_pct'] = (df['price_vs_avg'] / df['avg_price'] * 100).fillna(0)
    
    # Is this the cheapest/most expensive?
    df['is_cheapest'] = (df['price_numeric'] == df['min_price']).astype(int)
    df['is_most_expensive'] = (df['price_numeric'] == df['max_price']).astype(int)
    
    # Price percentile within group (0-100)
    df['price_percentile'] = grouped['price_numeric'].rank(pct=True).fillna(0.5) * 100
    
    # Store rank for this product (1 = cheapest)
    df['store_rank'] = grouped['price_numeric'].rank(method='min').fillna(1).astype(int)
    
    print(f"[+] Generated cross-store features")
    
    # Print summary
    multi_store = df[df['num_stores'] >= 2]
    print(f"\n[i] Products available in multiple stores: {len(multi_store)}")
    print(f"[i] Average price range across stores: {price_stats['price_range'].mean():.2f} PKR")
    print(f"[i] Average price variance: {price_stats['price_range_pct'].mean():.2f}%")
    
    return df


def generate_store_competitiveness(df):
    """
    Calculate store-level pricing competitiveness.
    """
    print("\n[i] Calculating store competitiveness metrics...")
    
    # For products available in multiple stores, how often is each store cheapest?
    multi_store_products = df[df['num_stores'] >= 2].copy()
    
    if len(multi_store_products) > 0:
        store_stats = multi_store_products.groupby('site').agg({
            'is_cheapest': 'sum',
            'price_vs_avg_pct': 'mean',
            'product_group_id': 'count'
        }).reset_index()
        
        store_stats.columns = ['site', 'times_cheapest', 'avg_price_vs_market', 'products_compared']
        store_stats['cheapest_rate'] = (store_stats['times_cheapest'] / store_stats['products_compared'] * 100)
        
        print("\n=== Store Competitiveness Summary ===")
        print(store_stats.to_string(index=False))
        
        # Merge store competitiveness back to main df
        df = df.merge(
            store_stats[['site', 'avg_price_vs_market', 'cheapest_rate']],
            on='site',
            how='left',
            suffixes=('', '_store')
        )
    
    return df


def save_unified_dataset(df):
    """Save the unified dataset with all features."""
    print(f"\n[i] Saving unified dataset to {OUTPUT_FILE}...")
    
    # Select relevant columns
    output_cols = [
        'product_group_id', 'site', 'category', 'title', 'normalized_name',
        'price', 'price_numeric', 'brand', 'size_value', 'size_unit', 
        'normalized_quantity', 'pack_count', 'total_volume',
        'min_price', 'max_price', 'avg_price', 'median_price', 'std_price',
        'price_range', 'price_range_pct', 'num_stores',
        'price_vs_avg', 'price_vs_avg_pct', 'is_cheapest', 'is_most_expensive',
        'price_percentile', 'store_rank', 'avg_price_vs_market', 'cheapest_rate',
        'brand_category', 'unit_price', 'per_unit_price'
    ]
    
    # Only include columns that exist
    available_cols = [col for col in output_cols if col in df.columns]
    
    df_output = df[available_cols].copy()
    df_output.to_csv(OUTPUT_FILE, index=False)
    
    print(f"[+] Saved {len(df_output)} products to {os.path.basename(OUTPUT_FILE)}")
    print(f"[i] Columns: {len(available_cols)}")
    
    return OUTPUT_FILE


def main():
    """Main pipeline."""
    print("="*60)
    print("Fuzzy Product Matching & Cross-Store Price Comparison")
    print("="*60)
    
    # Load all products
    df = load_all_products()
    if df is None:
        return
    
    # Normalize product names
    df = create_normalized_names(df)
    
    # Fuzzy match products across stores
    df = fuzzy_match_products(df, similarity_threshold=85)
    
    # Generate cross-store price features
    df = generate_cross_store_features(df)
    
    # Calculate store competitiveness
    df = generate_store_competitiveness(df)
    
    # Save unified dataset
    output_file = save_unified_dataset(df)
    
    print("\n" + "="*60)
    print("[✓] Fuzzy matching complete!")
    print(f"[✓] Unified dataset: {output_file}")
    print("="*60)
    
    print("\n[i] Next steps:")
    print("    1. Review unified dataset")
    print("    2. Train model with new cross-store features")
    print("    3. Model should now learn store pricing patterns")


if __name__ == '__main__':
    main()
