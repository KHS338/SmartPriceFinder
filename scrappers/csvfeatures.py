#!/usr/bin/env python3
"""
Extract features from product CSVs without filtering or dropping rows.
Operates on all CSV files in fullproductcsvs/ folder and creates *_features.csv versions.

Features extracted:
- size_value, size_unit, pack_count
- normalized_quantity (L→ml, kg→g)
- unit_price, unit_price_unit, per_unit_price
- notes (informational flags)

Does NOT:
- Drop any rows
- Filter by unit type
- Overwrite original files

Usage: python csvfeatures.py
"""

import re
import os
import glob
import pandas as pd

# Define the directory containing CSV files
CSV_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fullproductscsvs')

# Known brands in Pakistani grocery market
KNOWN_BRANDS = [
    # Dairy & Beverages
    'nestle', 'national', 'tapal', 'lipton', 'brookebond', 'olpers', 'nurpur', 'haleeb', 'shezan',
    # Food & Snacks
    'shan', 'knorr', 'maggi', 'mitchells', 'unilever', 'dalda', 'seasons', 'habib', 'kisan',
    # Personal Care
    'lux', 'lifebuoy', 'dove', 'pantene', 'head&shoulders', 'fair&lovely', 'garnier', 'loreal',
    # Household
    'surf', 'ariel', 'tide', 'comfort', 'harpic', 'vim', 'rin', 'persil',
    # Spices & Condiments
    'national', 'shan', 'mehran', 'laziza', 'ahmed',
    # Cooking oils
    'dalda', 'habib', 'kisan', 'sufi', 'golden', 'seasons',
    # Biscuits & Snacks
    'peek freans', 'gala', 'sooper', 'cocomo', 'rio', 'bisconni', 'novita', 'english biscuit',
    # Rice & Grains  
    'guard', 'super', 'supreme', 'kainat', 'basmati',
    # Additional
    'coca-cola', 'pepsi', 'sprite', 'fanta', 'mountain dew', 'aquafina', 'kinley',
    'colgate', 'closeup', 'sensodyne', 'pepsodent',
    'johnson', 'pampers', 'huggies', 'dettol', 'safeguard',
]


# PACK and SIZE patterns (case-insensitive)
PACK_PATTERNS = [
    r'\(pack of\s*(\d{1,3})\)',  # (Pack of 12) - highest priority
    r'\bpack of\s*(\d{1,3})\b',  # Pack of 12
    r'\bpk of\s*(\d{1,3})\b',    # Pk of 12
    r'\bpack\s*\((\d{1,3})\)',   # Pack (12)
    r'\b(\d{1,3})\s*pcs\b',      # 12 pcs
    r'\b(\d{1,3})\s*pc\b',       # 12 pc
    r'\b(\d{1,3})\s*pack\b',     # 12 pack (but will skip if followed by unit)
    r'\bpack\s*(\d{1,3})\b',     # Pack 12
    r'\bpk\s*(\d{1,3})\b',       # Pk 12
]

SIZE_PATTERNS = [
    (r'(\d+(?:[.,]\d+)?)\s*(?:ml|milli?lit(?:re|er)s?)\b', 'ml'),
    (r'(\d+(?:[.,]\d+)?)\s*ml\b', 'ml'),
    (r"(\d+(?:[.,]\d+)?)\s*(?:l\b|ltr\b|lt\b|litre\b|litres\b|liter\b|liters\b|ltrs\b)", 'l'),
    (r'(\d+(?:[.,]\d+)?)\s*(?:g\b|gm\b|grams?\b)\b', 'g'),
    (r'(\d+(?:[.,]\d+)?)\s*(?:kg\b|kgs\b|kilogram(?:s)?\b)', 'kg'),
    (r'(\d+)\s*(?:pcs?|pieces|pc\.)\b', 'pc'),
    (r'\b(\d+)\s*(?:pack|pk|pkt)\b', 'packnum'),
    (r'(\d+(?:[.,]\d+)?)(?:ml)\b', 'ml'),
    (r'(\d+(?:[.,]\d+)?)(?:l)\b', 'l'),
    (r'(\d+(?:[.,]\d+)?)(?:kg)\b', 'kg'),
]


def to_float(s):
    if s is None:
        return None
    s = str(s).replace(',', '.')
    try:
        return float(s)
    except:
        return None


def find_pack_count(title):
    title_l = title.lower()
    for pat in PACK_PATTERNS:
        m = re.search(pat, title_l, flags=re.IGNORECASE)
        if m:
            for g in m.groups():
                if g:
                    try:
                        v = int(g)
                        if v > 0:
                            # Check if this number is followed by a unit (ml, l, g, kg, etc.)
                            # to avoid matching "250" from "250 ml" as pack count
                            num_pos = m.start(1)  # position of the captured number
                            # Look ahead in original title to see if unit follows
                            check_snippet = title_l[num_pos:num_pos+20]
                            if re.search(r'^\d+\s*(?:ml|l|ltr|lt|g|gm|kg|kgs)\b', check_snippet):
                                continue  # Skip this match, it's a size not a pack
                            return v
                    except:
                        continue
    return None


def find_size(title):
    title_l = title.lower()
    for pat, unit in SIZE_PATTERNS:
        m = re.search(pat, title_l, flags=re.IGNORECASE)
        if m:
            val = m.group(1)
            num = to_float(val)
            if num is None:
                continue
            return num, unit
    # fallback: any numeric + known unit close by
    m = re.search(r'(\d+(?:[.,]\d+)?)\s*(ml|l|ltr|lt|g|kg|gm|pcs|pc)\b', title_l)
    if m:
        num = to_float(m.group(1))
        return num, m.group(2)
    return None, None


def parse_existing_quantity(qty_str):
    """
    Parse existing quantity column (e.g., from Rainbow scraper).
    Returns (size_value, size_unit) tuple.
    Examples: "1Kg" → (1, 'kg'), "500ml" → (500, 'ml')
    """
    if not isinstance(qty_str, str):
        qty_str = str(qty_str or '').strip()
    
    if not qty_str:
        return None, None
    
    # Try to match number + unit patterns
    for pat, unit in SIZE_PATTERNS:
        m = re.search(pat, qty_str.lower(), flags=re.IGNORECASE)
        if m:
            val = m.group(1)
            num = to_float(val)
            if num is not None:
                return num, unit
    
    return None, None


def extract_brand(title):
    """
    Extract brand name from product title.
    First checks known brands list, then falls back to first word as brand.
    """
    if not isinstance(title, str):
        title = str(title or '')
    
    title_lower = title.lower()
    
    # First, check for known brands in the list
    for brand in KNOWN_BRANDS:
        # Use word boundary to avoid partial matches
        pattern = r'\b' + re.escape(brand) + r'\b'
        if re.search(pattern, title_lower):
            return brand.title()  # Return with proper capitalization
    
    # If no known brand found, use first word as brand
    words = title.strip().split()
    if words:
        first_word = words[0].strip()
        # Clean up first word (remove special chars except letters and numbers)
        first_word = re.sub(r'[^a-zA-Z0-9&\-]', '', first_word)
        if first_word:
            return first_word.title()
    
    return 'Unknown'


def extract_from_title(title):
    """Extract features from product title."""
    if not isinstance(title, str):
        title = str(title or '')
    t = title.strip()

    size_val, size_unit = find_size(t)
    pack_count = find_pack_count(t)
    
    # Default pack_count to 0 if None
    if pack_count is None:
        pack_count = 0

    canonical_unit = None
    if size_unit:
        su = size_unit.lower()
        if su.startswith('l'):
            canonical_unit = 'l'
        elif su.startswith('ml'):
            canonical_unit = 'ml'
        elif su.startswith('g') and not su.startswith('kg'):
            canonical_unit = 'g'
        elif su.startswith('kg'):
            canonical_unit = 'kg'
        elif su in ('pc', 'packnum'):
            canonical_unit = 'pc'
        else:
            canonical_unit = su

    # Normalize to ml or g
    normalized_quantity = None
    
    if canonical_unit == 'ml':
        normalized_quantity = size_val
    elif canonical_unit == 'l' and size_val is not None:
        normalized_quantity = size_val * 1000  # liters to ml
    elif canonical_unit == 'g':
        normalized_quantity = size_val
    elif canonical_unit == 'kg' and size_val is not None:
        normalized_quantity = size_val * 1000  # kg to g

    notes = []
    if size_val is None and pack_count == 0:
        notes.append('no-size-no-pack')
    else:
        if size_val is None:
            notes.append('no-size')
        if pack_count == 0:
            notes.append('no-pack')

    return {
        'size_value': size_val,
        'size_unit': canonical_unit,
        'pack_count': pack_count,
        'normalized_quantity': normalized_quantity,
        'notes': ';'.join(notes) if notes else ''
    }


def process_csv(infile, output_dir):
    """Process a single CSV file and add feature columns."""
    print(f"\n[Processing] {os.path.basename(infile)}")
    
    df = pd.read_csv(infile, dtype=str).fillna('')
    
    if 'title' not in df.columns:
        print(f"[!] Skipping {infile}: no 'title' column found")
        return
    
    # Check if 'quantity' column already exists (e.g., from Rainbow scraper)
    has_quantity_col = 'quantity' in df.columns
    
    if has_quantity_col:
        print(f"[i] Found existing 'quantity' column, parsing it for size info")
        # Extract from existing quantity column
        qty_parsed = df['quantity'].apply(lambda q: pd.Series(parse_existing_quantity(q)))
        qty_parsed.columns = ['qty_size_value', 'qty_size_unit']
        
        # Extract from title as well
        title_features = df['title'].apply(lambda t: pd.Series(extract_from_title(t)))
        
        # Merge: prefer quantity column data, fallback to title data
        df['size_value'] = qty_parsed['qty_size_value'].where(
            qty_parsed['qty_size_value'].notna(), 
            title_features['size_value']
        )
        df['size_unit'] = qty_parsed['qty_size_unit'].where(
            qty_parsed['qty_size_unit'].notna(), 
            title_features['size_unit']
        )
        df['pack_count'] = title_features['pack_count']
        df['notes'] = title_features['notes']
        
    else:
        print(f"[i] No 'quantity' column found, extracting from title only")
        # Extract from title
        features = df['title'].apply(lambda t: pd.Series(extract_from_title(t)))
        df['size_value'] = features['size_value']
        df['size_unit'] = features['size_unit']
        df['pack_count'] = features['pack_count']
        df['notes'] = features['notes']
    
    # Calculate normalized_quantity based on size_value and size_unit
    df['normalized_quantity'] = None
    
    # Convert to numeric for calculation
    df['size_value_numeric'] = pd.to_numeric(df['size_value'], errors='coerce')
    
    # ml stays as-is
    ml_mask = df['size_unit'] == 'ml'
    df.loc[ml_mask, 'normalized_quantity'] = df.loc[ml_mask, 'size_value_numeric']
    
    # l to ml
    l_mask = df['size_unit'] == 'l'
    df.loc[l_mask, 'normalized_quantity'] = df.loc[l_mask, 'size_value_numeric'] * 1000
    
    # g stays as-is
    g_mask = df['size_unit'] == 'g'
    df.loc[g_mask, 'normalized_quantity'] = df.loc[g_mask, 'size_value_numeric']
    
    # kg to g
    kg_mask = df['size_unit'] == 'kg'
    df.loc[kg_mask, 'normalized_quantity'] = df.loc[kg_mask, 'size_value_numeric'] * 1000
    
    # Extract brand from title
    print(f"[i] Extracting brand names from titles...")
    df['brand'] = df['title'].apply(extract_brand)
    brands_found = (df['brand'] != 'Unknown').sum()
    print(f"[i] Identified brands in {brands_found}/{len(df)} products")
    
    # Create Size × Pack interaction (total_volume)
    df['pack_count_numeric'] = pd.to_numeric(df['pack_count'], errors='coerce').fillna(0)
    df['total_volume'] = df['size_value_numeric'] * df['pack_count_numeric'].replace(0, 1)
    print(f"[i] Created total_volume feature (size × pack)")
    
    # Create Brand × Category interaction
    if 'category' in df.columns:
        df['brand_category'] = df['brand'].astype(str) + '_' + df['category'].astype(str)
        print(f"[i] Created brand_category interaction feature")
    else:
        df['brand_category'] = df['brand'].astype(str)
        print(f"[i] No category column found, using brand only")
    
    # Calculate prices if 'price' column exists
    if 'price' in df.columns:
        df['price_numeric'] = pd.to_numeric(df['price'], errors='coerce')
        
        # unit_price: price per ml or price per g
        df['unit_price'] = None
        df['unit_price_unit'] = ''
        
        # For liquid products (ml, l)
        liquid_mask = (df['size_unit'].isin(['ml', 'l'])) & pd.notna(df['normalized_quantity']) & (df['normalized_quantity'] > 0) & pd.notna(df['price_numeric'])
        df.loc[liquid_mask, 'unit_price'] = df.loc[liquid_mask, 'price_numeric'] / df.loc[liquid_mask, 'normalized_quantity']
        df.loc[liquid_mask, 'unit_price_unit'] = 'PKR/ml'
        
        # For solid products (g, kg)
        solid_mask = (df['size_unit'].isin(['g', 'kg'])) & pd.notna(df['normalized_quantity']) & (df['normalized_quantity'] > 0) & pd.notna(df['price_numeric'])
        df.loc[solid_mask, 'unit_price'] = df.loc[solid_mask, 'price_numeric'] / df.loc[solid_mask, 'normalized_quantity']
        df.loc[solid_mask, 'unit_price_unit'] = 'PKR/g'
        
        # per_unit_price: price per individual item in pack
        df['per_unit_price'] = None
        
        # For items with pack_count > 0: divide price by pack_count
        pack_mask = (df['pack_count_numeric'] > 0) & pd.notna(df['price_numeric'])
        df.loc[pack_mask, 'per_unit_price'] = df.loc[pack_mask, 'price_numeric'] / df.loc[pack_mask, 'pack_count_numeric']
        
        # For items with pack_count = 0: use full price
        no_pack_mask = (df['pack_count_numeric'] == 0) & pd.notna(df['price_numeric'])
        df.loc[no_pack_mask, 'per_unit_price'] = df.loc[no_pack_mask, 'price_numeric']
        
        # Drop temporary columns
        df = df.drop(columns=['price_numeric', 'pack_count_numeric', 'size_value_numeric'])
    else:
        df = df.drop(columns=['size_value_numeric', 'pack_count_numeric'])
    
    # Generate output filename in the same directory as input
    base_name = os.path.basename(infile)
    name_without_ext = os.path.splitext(base_name)[0]
    outfile = os.path.join(output_dir, f"{name_without_ext}_features.csv")
    
    # Save to new file (don't overwrite original)
    df.to_csv(outfile, index=False)
    print(f"[+] Saved {len(df)} rows to {os.path.basename(outfile)}")
    
    # Print statistics
    total = len(df)
    has_size = (pd.notna(df['size_value']) & (df['size_value'] != '')).sum()
    has_pack = (pd.to_numeric(df['pack_count'], errors='coerce') > 0).sum()
    has_brand = (df['brand'] != 'Unknown').sum()
    
    print(f"[i] Statistics: {has_size}/{total} with size, {has_pack}/{total} with pack count, {has_brand}/{total} with brand")
    
    if 'unit_price' in df.columns:
        valid_unit_prices = df['unit_price'].notna().sum()
        valid_per_unit_prices = df['per_unit_price'].notna().sum()
        print(f"[i] Calculated unit_price for {valid_unit_prices} products")
        print(f"[i] Calculated per_unit_price for {valid_per_unit_prices} products")


def main():
    """Process all CSV files in fullproductcsvs/ directory."""
    # Check if directory exists
    if not os.path.exists(CSV_DIR):
        print(f"[!] Directory not found: {CSV_DIR}")
        print(f"[!] Please create the 'fullproductcsvs' folder in the parent directory")
        return
    
    # Find all CSV files in the directory
    search_pattern = os.path.join(CSV_DIR, "*.csv")
    csv_files = glob.glob(search_pattern)
    
    # Filter out files that already have '_features' in the name
    csv_files = [f for f in csv_files if '_features' not in os.path.basename(f)]
    
    if not csv_files:
        print(f"[!] No CSV files found in {CSV_DIR}")
        return
    
    print(f"[i] Found {len(csv_files)} CSV files to process in {os.path.basename(CSV_DIR)}:")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")
    
    for csv_file in csv_files:
        try:
            process_csv(csv_file, CSV_DIR)
        except Exception as e:
            print(f"[!] Error processing {os.path.basename(csv_file)}: {e}")
            continue
    
    print("\n[✓] All files processed!")


if __name__ == '__main__':
    main()
