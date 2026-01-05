

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import re


class BestDealModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.price_stats = {}
        self.brand_stats = {}
        self.category_stats = {}
        self.site_stats = {}
        self.outlier_detector = IsolationForest(contamination=0.1, random_state=42)
        self.products_df = None
        
    def extract_quantity_value(self, title):
        """Extract numeric quantity from title for unit price calculation"""
        if pd.isna(title):
            return 1.0
        
        patterns = [
            (r'(\d+\.?\d*)\s*l(?:iter)?s?', 1000),  # liters to ml
            (r'(\d+\.?\d*)\s*ml', 1),               # ml
            (r'(\d+\.?\d*)\s*kg', 1000),            # kg to g
            (r'(\d+\.?\d*)\s*g(?:ram)?s?', 1),      # grams
            (r'(\d+)\s*pack', 1),                   # pack
            (r'(\d+)\s*pcs?', 1),                   # pieces
        ]
        
        title_lower = str(title).lower()
        for pattern, multiplier in patterns:
            match = re.search(pattern, title_lower)
            if match:
                value = float(match.group(1))
                return value * multiplier
        
        return 1.0
    
    def calculate_value_indicators(self, title):
        """Calculate value score based on keywords in title"""
        if pd.isna(title):
            return 0
        
        title_lower = str(title).lower()
        
        # Positive indicators (premium/quality)
        positive_keywords = ['premium', 'organic', 'natural', 'pure', 'fresh', 
                           'imported', 'quality', 'best', 'deluxe']
        # Negative indicators (budget/economy)
        negative_keywords = ['economy', 'value', 'basic', 'budget', 'saver']
        
        score = 0
        for keyword in positive_keywords:
            if keyword in title_lower:
                score += 1
        
        for keyword in negative_keywords:
            if keyword in title_lower:
                score -= 1
        
        return score
    
    def build_statistics(self, df):
        """Build price statistics for brands, categories, and sites"""
        print("\n[Statistics] Building price statistics...")
        
        # Overall price statistics
        self.price_stats = {
            'mean': df['price'].mean(),
            'std': df['price'].std(),
            'min': df['price'].min(),
            'max': df['price'].max(),
            'median': df['price'].median()
        }
        
        # Brand-level statistics
        for brand in df['brand'].dropna().unique():
            if brand and brand.strip():
                brand_data = df[df['brand'] == brand]['price']
                self.brand_stats[brand] = {
                    'mean': brand_data.mean(),
                    'std': brand_data.std(),
                    'count': len(brand_data)
                }
        
        # Category-level statistics
        for category in df['category'].dropna().unique():
            if category and category.strip():
                cat_data = df[df['category'] == category]['price']
                self.category_stats[category] = {
                    'mean': cat_data.mean(),
                    'std': cat_data.std(),
                    'count': len(cat_data)
                }
        
        # Site-level statistics
        for site in df['site'].dropna().unique():
            if site and site.strip():
                site_data = df[df['site'] == site]['price']
                self.site_stats[site] = {
                    'mean': site_data.mean(),
                    'std': site_data.std(),
                    'count': len(site_data)
                }
        
        print(f"  ✓ {len(self.brand_stats)} brands")
        print(f"  ✓ {len(self.category_stats)} categories")
        print(f"  ✓ {len(self.site_stats)} sites")
    
    def calculate_deal_score(self, row):
        """Calculate deal score for a single product (0-100)"""
        score_components = []
        
        # 1. Price vs Overall Average (40%)
        price_vs_avg = (self.price_stats['mean'] - row['price']) / (self.price_stats['std'] + 0.01)
        score_components.append(price_vs_avg * 0.4)
        
        # 2. Price vs Brand Average (20%)
        brand = row.get('brand')
        if brand and brand in self.brand_stats:
            brand_mean = self.brand_stats[brand]['mean']
            brand_std = self.brand_stats[brand]['std']
            price_vs_brand = (brand_mean - row['price']) / (brand_std + 0.01)
            score_components.append(price_vs_brand * 0.2)
        
        # 3. Price vs Category Average (20%)
        category = row.get('category')
        if category and category in self.category_stats:
            cat_mean = self.category_stats[category]['mean']
            cat_std = self.category_stats[category]['std']
            price_vs_cat = (cat_mean - row['price']) / (cat_std + 0.01)
            score_components.append(price_vs_cat * 0.2)
        
        # 4. Unit Price Efficiency (10%)
        if 'unit_price' in row and pd.notna(row['unit_price']) and row['unit_price'] > 0:
            # Lower unit price is better
            max_unit_price = self.products_df['unit_price'].max()
            unit_price_score = 1 - (row['unit_price'] / max_unit_price)
            score_components.append(unit_price_score * 0.1)
        
        # 5. Value Indicators (10%)
        value_score = row.get('value_score', 0) / 5  # Normalize to 0-1
        score_components.append(value_score * 0.1)
        
        # Combine all components
        total_score = sum(score_components)
        
        return total_score
    
    def train(self, csv_files):
        """Train the best deal model on multiple CSV files"""
        print("\n" + "="*60)
        print("Training Best Deal Identification Model")
        print("="*60)
        
        # Load and combine all CSV files
        dfs = []
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                print(f"\n[Loading] {csv_file}")
                df = pd.read_csv(csv_file)
                dfs.append(df)
                print(f"  ✓ Loaded {len(df)} products")
            else:
                print(f"  ✗ File not found: {csv_file}")
        
        if not dfs:
            raise ValueError("No CSV files loaded!")
        
        # Combine all dataframes
        self.products_df = pd.concat(dfs, ignore_index=True)
        print(f"\n[Total] {len(self.products_df)} products from {len(dfs)} sources")
        
        # Remove products without prices
        self.products_df = self.products_df[self.products_df['price'].notna()].copy()
        print(f"[Filtered] {len(self.products_df)} products with valid prices")
        
        # Extract quantity values for unit price calculation
        print("\n[Processing] Extracting quantity values...")
        if 'normalized_quantity' not in self.products_df.columns:
            self.products_df['quantity_value'] = self.products_df['title'].apply(
                self.extract_quantity_value
            )
        else:
            self.products_df['quantity_value'] = self.products_df['normalized_quantity'].fillna(
                self.products_df['title'].apply(self.extract_quantity_value)
            )
        
        # Calculate price per unit
        self.products_df['price_per_unit'] = (
            self.products_df['price'] / self.products_df['quantity_value']
        )
        
        # Calculate value indicators
        print("[Processing] Calculating value indicators...")
        self.products_df['value_score'] = self.products_df['title'].apply(
            self.calculate_value_indicators
        )
        
        # Build statistics
        self.build_statistics(self.products_df)
        
        # Calculate deal scores
        print("\n[Training] Calculating deal scores...")
        self.products_df['raw_deal_score'] = self.products_df.apply(
            self.calculate_deal_score, axis=1
        )
        
        # Normalize scores to 0-100
        min_score = self.products_df['raw_deal_score'].min()
        max_score = self.products_df['raw_deal_score'].max()
        
        self.products_df['deal_score'] = 100 * (
            (self.products_df['raw_deal_score'] - min_score) / 
            (max_score - min_score + 0.01)
        )
        
        print(f"  ✓ Deal scores calculated")
        print(f"  ✓ Score range: {self.products_df['deal_score'].min():.1f} - {self.products_df['deal_score'].max():.1f}")
        
        # Train outlier detector on price features
        print("\n[Training] Training outlier detection...")
        price_features = self.products_df[['price', 'price_per_unit']].fillna(0)
        self.outlier_detector.fit(price_features)
        print(f"  ✓ Outlier detector trained")
        
        print("\n" + "="*60)
        print("✅ Model Training Complete!")
        print("="*60)
        
        return self
    
    def predict_deal_score(self, product_data):
        """Predict deal score for a new product"""
        # This will be used by analyze_products.py
        score = self.calculate_deal_score(product_data)
        
        # Normalize using learned min/max
        min_score = self.products_df['raw_deal_score'].min()
        max_score = self.products_df['raw_deal_score'].max()
        
        normalized_score = 100 * (score - min_score) / (max_score - min_score + 0.01)
        return max(0, min(100, normalized_score))
    
    def get_top_deals(self, n=20):
        """Get top N deals from training data"""
        top_deals = self.products_df.nlargest(n, 'deal_score')
        
        results = []
        for _, product in top_deals.iterrows():
            results.append({
                'title': product['title'],
                'brand': product.get('brand', 'N/A'),
                'category': product.get('category', 'N/A'),
                'price': product.get('price', 0),
                'site': product.get('site', 'N/A'),
                'deal_score': product['deal_score'],
                'price_per_unit': product.get('price_per_unit', 0)
            })
        
        return results
    
    def save(self, filepath):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"\n[Saved] Model saved to {filepath}")


def main():
    # Define paths to feature CSV files
    csv_dir = Path('fullproductscsvs')
    feature_csvs = [
        csv_dir / 'alfatah_all_products_features.csv',
        csv_dir / 'imtiaz_all_products_features.csv',
        csv_dir / 'naheed_groceries_products_full_features.csv',
        csv_dir / 'rainbow_products_features.csv'
    ]
    
    # Filter only existing files
    existing_csvs = [str(csv) for csv in feature_csvs if csv.exists()]
    
    if not existing_csvs:
        print("Error: No feature CSV files found in fullproductscsvs/")
        return
    
    print(f"\n[Found] {len(existing_csvs)} feature CSV files")
    
    # Train the model
    model = BestDealModel()
    model.train(existing_csvs)
    
    # Save the model
    model_path = 'models/best_deal_model.pkl'
    model.save(model_path)
    
    # Test the model with top deals
    print("\n" + "="*60)
    print("Testing Model - Top 10 Best Deals")
    print("="*60)
    
    top_deals = model.get_top_deals(n=10)
    for i, deal in enumerate(top_deals, 1):
        print(f"\n{i}. {deal['title']}")
        print(f"   Site: {deal['site']} | Brand: {deal['brand']}")
        print(f"   Price: Rs. {deal['price']:.2f} | Deal Score: {deal['deal_score']:.1f}/100")
        if deal['price_per_unit'] > 0:
            print(f"   Price/Unit: Rs. {deal['price_per_unit']:.2f}")
    
    print("\n" + "="*60)
    print("✅ Model Training and Testing Complete!")
    print("="*60)
    print(f"\nModel saved to: {model_path}")
    print("Use this model with analyze_products.py for deal identification")
    
    # Show statistics
    print("\n" + "="*60)
    print("Model Statistics")
    print("="*60)
    print(f"Total products analyzed: {len(model.products_df)}")
    print(f"Average price: Rs. {model.price_stats['mean']:.2f}")
    print(f"Price range: Rs. {model.price_stats['min']:.2f} - Rs. {model.price_stats['max']:.2f}")
    print(f"Brands tracked: {len(model.brand_stats)}")
    print(f"Categories tracked: {len(model.category_stats)}")
    print(f"Sites analyzed: {len(model.site_stats)}")


if __name__ == "__main__":
    main()
