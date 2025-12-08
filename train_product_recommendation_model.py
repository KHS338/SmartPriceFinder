

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import re


class ProductRecommendationModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True
        )
        self.scaler = StandardScaler()
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.products_df = None
        self.brand_groups = {}
        self.category_groups = {}
        
    def preprocess_text(self, text):
        """Clean and preprocess product text"""
        if pd.isna(text):
            return ""
        # Convert to lowercase and remove special characters
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return text
    
    def extract_features(self, df):
        """Extract and combine features for similarity calculation"""
        # Combine title, brand, and category for rich feature representation
        df['combined_features'] = (
            df['title'].fillna('').astype(str) + ' ' +
            df['brand'].fillna('').astype(str) + ' ' +
            df['category'].fillna('').astype(str) + ' ' +
            df['brand_category'].fillna('').astype(str)
        )
        
        # Preprocess combined features
        df['processed_features'] = df['combined_features'].apply(self.preprocess_text)
        
        return df
    
    def build_brand_groups(self, df):
        """Group products by brand for brand-based recommendations"""
        for brand in df['brand'].dropna().unique():
            if brand and brand.strip():
                self.brand_groups[brand] = df[df['brand'] == brand].index.tolist()
    
    def build_category_groups(self, df):
        """Group products by category for category-based recommendations"""
        for category in df['category'].dropna().unique():
            if category and category.strip():
                self.category_groups[category] = df[df['category'] == category].index.tolist()
    
    def train(self, csv_files):
        """Train the recommendation model on multiple CSV files"""
        print("\n" + "="*60)
        print("Training Product Recommendation Model")
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
        
        # Extract features
        print("\n[Processing] Extracting features...")
        self.products_df = self.extract_features(self.products_df)
        
        # Build brand and category groups
        print("[Processing] Building brand groups...")
        self.build_brand_groups(self.products_df)
        print(f"  ✓ {len(self.brand_groups)} unique brands")
        
        print("[Processing] Building category groups...")
        self.build_category_groups(self.products_df)
        print(f"  ✓ {len(self.category_groups)} unique categories")
        
        # Create TF-IDF matrix
        print("\n[Training] Creating TF-IDF matrix...")
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.products_df['processed_features']
        )
        print(f"  ✓ Matrix shape: {self.tfidf_matrix.shape}")
        
        # Calculate similarity matrix (only for a sample to save memory)
        print("[Training] Calculating similarity matrix...")
        # Use sparse matrix for memory efficiency
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix, dense_output=False)
        print(f"  ✓ Similarity matrix computed")
        
        print("\n" + "="*60)
        print("✅ Model Training Complete!")
        print("="*60)
        
        return self
    
    def get_similar_products(self, product_idx, top_n=10):
        """Get similar products based on cosine similarity"""
        if self.similarity_matrix is None:
            return []
        
        # Get similarity scores for the product
        sim_scores = self.similarity_matrix[product_idx].toarray().flatten()
        
        # Get top N similar products (excluding itself)
        similar_indices = sim_scores.argsort()[-top_n-1:-1][::-1]
        
        results = []
        for idx in similar_indices:
            if idx != product_idx:
                results.append({
                    'index': int(idx),
                    'title': self.products_df.iloc[idx]['title'],
                    'brand': self.products_df.iloc[idx].get('brand', 'N/A'),
                    'category': self.products_df.iloc[idx].get('category', 'N/A'),
                    'price': self.products_df.iloc[idx].get('price', 0),
                    'site': self.products_df.iloc[idx].get('site', 'N/A'),
                    'similarity': float(sim_scores[idx])
                })
        
        return results
    
    def get_brand_recommendations(self, brand, top_n=10):
        """Get top products from a specific brand"""
        if brand not in self.brand_groups:
            return []
        
        indices = self.brand_groups[brand]
        products = self.products_df.iloc[indices]
        
        # Sort by price (cheapest first)
        products = products.sort_values('price', ascending=True)
        
        results = []
        for _, product in products.head(top_n).iterrows():
            results.append({
                'title': product['title'],
                'brand': product.get('brand', 'N/A'),
                'category': product.get('category', 'N/A'),
                'price': product.get('price', 0),
                'site': product.get('site', 'N/A')
            })
        
        return results
    
    def get_category_recommendations(self, category, top_n=10):
        """Get top products from a specific category"""
        if category not in self.category_groups:
            return []
        
        indices = self.category_groups[category]
        products = self.products_df.iloc[indices]
        
        # Sort by price (cheapest first)
        products = products.sort_values('price', ascending=True)
        
        results = []
        for _, product in products.head(top_n).iterrows():
            results.append({
                'title': product['title'],
                'brand': product.get('brand', 'N/A'),
                'category': product.get('category', 'N/A'),
                'price': product.get('price', 0),
                'site': product.get('site', 'N/A')
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
    model = ProductRecommendationModel()
    model.train(existing_csvs)
    
    # Save the model
    model_path = 'models/product_recommendation_model.pkl'
    model.save(model_path)
    
    # Test the model with a sample
    print("\n" + "="*60)
    print("Testing Model - Sample Recommendations")
    print("="*60)
    
    # Test similarity-based recommendations
    if len(model.products_df) > 0:
        sample_idx = 0
        sample_product = model.products_df.iloc[sample_idx]
        print(f"\n[Sample Product]")
        print(f"  Title: {sample_product['title']}")
        print(f"  Brand: {sample_product.get('brand', 'N/A')}")
        print(f"  Category: {sample_product.get('category', 'N/A')}")
        print(f"  Price: Rs. {sample_product.get('price', 0):.2f}")
        
        print(f"\n[Similar Products (Top 5)]")
        similar = model.get_similar_products(sample_idx, top_n=5)
        for i, prod in enumerate(similar, 1):
            print(f"\n  {i}. {prod['title']}")
            print(f"     Brand: {prod['brand']} | Price: Rs. {prod['price']:.2f}")
            print(f"     Similarity: {prod['similarity']:.3f}")
    
    print("\n" + "="*60)
    print("✅ Model Training and Testing Complete!")
    print("="*60)
    print(f"\nModel saved to: {model_path}")
    print("Use this model with analyze_products.py for recommendations")


if __name__ == "__main__":
    main()
