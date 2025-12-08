#!/usr/bin/env python3
"""
Extract quantities, pack counts, and calculate prices from product titles

Usage: python csvqty.py <input.csv>
"""

import re
import sys
import os
import pandas as pd

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
    'coca-cola', 'pepsi', 'sprite', 'fanta', 'mountain dew', '7up', 'aquafina', 'kinley',
    'colgate', 'closeup', 'sensodyne', 'pepsodent',
    'johnson', 'pampers', 'huggies', 'dettol', 'safeguard',
]

# PACK and SIZE patterns (case-insensitive)
# Reordered to prioritize explicit pack patterns and avoid matching size numbers
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
    # fallback: patterns like "(Pack of 12)" already covered, but also catch "pack12" etc.
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
    # ensure string
    if not isinstance(title, str):
        title = str(title or '')
    t = title.strip()

    size_val, size_unit = find_size(t)
    pack_count = find_pack_count(t)
    
    # Default pack_count to 0 if None to avoid empty cells
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

    # Normalize to ml or g - single column
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


def process_csv(infile):
    df = pd.read_csv(infile, dtype=str).fillna('')
    title_col = 'title'
    if title_col not in df.columns:
        raise ValueError(f"Title column '{title_col}' not in CSV. Columns: {df.columns.tolist()}")

    extras = df[title_col].apply(lambda t: pd.Series(extract_from_title(t)))

    # Only add the extracted columns to the original df (preserve other original columns)
    out = df.copy()
    out['size_value'] = extras['size_value']
    out['size_unit'] = extras['size_unit']
    out['pack_count'] = extras['pack_count']
    out['normalized_quantity'] = extras['normalized_quantity']
    out['notes'] = extras['notes']
    
    # Extract brand from title
    out['brand'] = out[title_col].apply(extract_brand)
    print(f"[i] Extracted brands for {len(out)} products")
    
    # Add brand_category column (combination of brand and a generic category based on product type)
    # For now, use 'Unknown' as category - can be enhanced later
    out['category'] = 'Grocery'  # Default category
    out['brand_category'] = out['brand'] + '_' + out['category']
    print(f"[i] Created brand_category column")
    
    # Drop products with no quantity information
    before = len(out)
    out = out[pd.notna(out['size_value']) & (out['size_value'] != '')].copy()
    dropped = before - len(out)
    if dropped > 0:
        print(f"[i] Dropped {dropped} products with no quantity information")
    
    if len(out) == 0:
        print("[!] Warning: All products dropped. No products with quantity information.")
        return
    
    # Detect majority unit type (liquid vs solid)
    liquid_count = out['size_unit'].isin(['ml', 'l']).sum()
    solid_count = out['size_unit'].isin(['g', 'kg']).sum()
    
    print(f"[i] Unit distribution: {liquid_count} liquid (ml/L), {solid_count} solid (g/kg)")
    
    # Drop minority unit type
    if liquid_count > 0 and solid_count > 0:
        before = len(out)
        if liquid_count > solid_count:
            # Keep liquids, drop solids
            out = out[out['size_unit'].isin(['ml', 'l'])].copy()
            dropped_minority = before - len(out)
            print(f"[i] Majority is liquid units. Dropped {dropped_minority} solid unit products")
        else:
            # Keep solids, drop liquids
            out = out[out['size_unit'].isin(['g', 'kg'])].copy()
            dropped_minority = before - len(out)
            print(f"[i] Majority is solid units. Dropped {dropped_minority} liquid unit products")
    
    if len(out) == 0:
        print("[!] Warning: All products dropped after filtering. Cannot proceed.")
        return
    
    # Calculate unit prices if price column exists
    if 'price' in out.columns:
        # Convert price to numeric, handling empty strings and non-numeric values
        out['price_numeric'] = pd.to_numeric(out['price'], errors='coerce')
        
        # unit_price: price per ml or price per g
        out['unit_price'] = None
        out['unit_price_unit'] = ''
        
        # Determine unit based on size_unit column
        # For liquid products (ml, l)
        liquid_mask = (out['size_unit'].isin(['ml', 'l'])) & pd.notna(out['normalized_quantity']) & (out['normalized_quantity'] > 0) & pd.notna(out['price_numeric'])
        out.loc[liquid_mask, 'unit_price'] = out.loc[liquid_mask, 'price_numeric'] / out.loc[liquid_mask, 'normalized_quantity']
        out.loc[liquid_mask, 'unit_price_unit'] = 'PKR/ml'
        
        # For solid products (g, kg)
        solid_mask = (out['size_unit'].isin(['g', 'kg'])) & pd.notna(out['normalized_quantity']) & (out['normalized_quantity'] > 0) & pd.notna(out['price_numeric'])
        out.loc[solid_mask, 'unit_price'] = out.loc[solid_mask, 'price_numeric'] / out.loc[solid_mask, 'normalized_quantity']
        out.loc[solid_mask, 'unit_price_unit'] = 'PKR/g'
        
        # per_unit_price: price per individual item in pack (or full price if no pack)
        out['per_unit_price'] = None
        
        # For items with pack_count > 0: divide price by pack_count
        pack_mask = (out['pack_count'].astype(float) > 0) & pd.notna(out['price_numeric'])
        out.loc[pack_mask, 'per_unit_price'] = out.loc[pack_mask, 'price_numeric'] / out.loc[pack_mask, 'pack_count'].astype(float)
        
        # For items with pack_count = 0: use full price
        no_pack_mask = (out['pack_count'].astype(float) == 0) & pd.notna(out['price_numeric'])
        out.loc[no_pack_mask, 'per_unit_price'] = out.loc[no_pack_mask, 'price_numeric']
        
        # Drop the temporary price_numeric column
        out = out.drop(columns=['price_numeric'])

    base, ext = os.path.splitext(infile)
    outfile = f"{base}_qtyfixed.csv"
    
    # Check if file exists and inform user
    if os.path.exists(outfile):
        print(f"[i] Overwriting existing file: {outfile}")
    
    out.to_csv(outfile, index=False)
    print(f"[+] Written {len(out)} rows to {outfile}")
    
    # Print summary statistics
    if 'unit_price' in out.columns:
        valid_unit_prices = out['unit_price'].notna().sum()
        valid_per_unit_prices = out['per_unit_price'].notna().sum()
        print(f"[i] Calculated unit_price for {valid_unit_prices} products")
        print(f"[i] Calculated per_unit_price for {valid_per_unit_prices} products")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python extract_qty.py input.csv')
        sys.exit(1)
    infile = sys.argv[1]
    process_csv(infile)
