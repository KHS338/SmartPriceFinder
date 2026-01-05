#!/usr/bin/env python3
"""
Interactive price comparison dashboard

Usage: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
import sys
import subprocess
import time

import sys
import subprocess
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scrappers'))

st.set_page_config(
    page_title="Smart Price Finder",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .best-deal {
        background-color: #d4edda;
        border: 2px solid #28a745;
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    
    numeric_cols = ['price', 'size_value', 'pack_count', 'normalized_quantity',
                    'normalized_volume_ml', 'normalized_weight_g', 'unit_price', 'per_unit_price',
                    'deal_score', 'price_percentile', 'value_score', 'store_rank', 'ml_predicted_score']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if 'normalized_quantity' not in df.columns and 'normalized_volume_ml' in df.columns:
        df['normalized_quantity'] = df['normalized_volume_ml']
    
    return df

def find_csv_files():
    files = list(Path('.').glob('*_analyzed.csv'))
    if not files:
        files = list(Path('./scrappers').glob('*_analyzed.csv'))
    
    if not files:
        files = list(Path('.').glob('*_qtyfixed.csv'))
    if not files:
        files = list(Path('./scrappers').glob('*_qtyfixed.csv'))
    
    return [str(f) for f in files]

def run_ml_analysis(qtyfixed_file, product_name):
    """Run ML analysis and display results with visualizations"""
    
    st.markdown("---")
    st.markdown("## 🤖 ML Analysis Panel")
    
    with st.spinner("🔄 Running ML models..."):
        # Load the qtyfixed CSV
        df = pd.read_csv(qtyfixed_file)
        
        # Determine correct working directory
        if 'scrappers' in qtyfixed_file:
            working_dir = 'scrappers'
        else:
            working_dir = '.'
        
        # Run the models
        if working_dir == 'scrappers':
            use_models_script = 'use_models_dashboard.py'
        else:
            use_models_script = os.path.join('scrappers', 'use_models_dashboard.py')
        
        csv_arg = os.path.basename(qtyfixed_file)
        
        models_cmd = [sys.executable, use_models_script, csv_arg]
        models_process = subprocess.run(
            models_cmd,
            capture_output=True,
            text=True,
            cwd=working_dir
        )
        
        if models_process.stdout:
            print(f"\n[ML Models OUTPUT]\n{models_process.stdout}")
            output_lines = models_process.stdout.split('\n')
            
            # Parse recommendations and deals data
            recommendations = []
            deals = []
            
            # Extract recommendations
            in_rec = False
            current_query = None
            for line in output_lines:
                if 'PRODUCT RECOMMENDATIONS' in line:
                    in_rec = True
                elif 'BEST DEAL IDENTIFICATION' in line:
                    in_rec = False
                elif in_rec:
                    if 'Query:' in line:
                        current_query = line.split('Query:')[1].strip() if 'Query:' in line else None
                    elif current_query and 'similarity:' in line:
                        recommendations.append({'query': current_query, 'line': line.strip()})
            
            # Extract deals
            in_deals = False
            current_deal = {}
            for line in output_lines:
                if 'BEST DEAL IDENTIFICATION' in line:
                    in_deals = True
                elif '[OK] ANALYSIS COMPLETE' in line:
                    break
                elif in_deals and line.strip():
                    if line.strip()[0:2] in ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.'] or line.strip()[0:3] == '10.':
                        if current_deal:
                            deals.append(current_deal)
                        current_deal = {'product': line.strip()}
                    elif 'Site:' in line:
                        current_deal['site'] = line.split('Site:')[1].strip()
                    elif 'Price:' in line:
                        current_deal['price'] = line.split('Price:')[1].strip()
                    elif 'Deal Score:' in line:
                        score = line.split('Deal Score:')[1].strip()
                        current_deal['score'] = float(score.split('/')[0])
            
            if current_deal:
                deals.append(current_deal)
            
            # Create two columns for layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### ⭐ Product Recommendations")
                st.markdown("Similar products based on ML training data")
                
                # Display recommendations
                rec_count = 0
                current_q = None
                for line in output_lines:
                    if 'PRODUCT RECOMMENDATIONS' in line:
                        in_rec = True
                    elif 'BEST DEAL IDENTIFICATION' in line:
                        break
                    elif 'Query:' in line:
                        current_q = line.split('Query:')[1].strip() if 'Query:' in line else None
                        if current_q:
                            st.markdown(f"**🔍 {current_q}**")
                            rec_count += 1
                    elif current_q and 'similarity:' in line and 'Alfatah' in line or 'Carrefour' in line or 'Imtiaz' in line or 'Naheed' in line or 'Rainbow' in line:
                        parts = line.strip().split('-')
                        if len(parts) >= 2:
                            product = parts[0].strip()
                            rest = '-'.join(parts[1:])
                            if 'similarity:' in rest:
                                site_price = rest.split('(similarity:')[0].strip()
                                similarity = rest.split('similarity:')[1].strip().replace(')', '')
                                st.markdown(f"  • {product}")
                                st.markdown(f"    *{site_price}* - Similarity: `{similarity}`")
                
                if rec_count == 0:
                    st.info("No recommendations found")
            
            with col2:
                st.markdown("### 🏆 Best Deals (ML Scored)")
                st.markdown("Top deals identified by ML model")
                
                # Display top 5 deals with metrics
                if deals:
                    for i, deal in enumerate(deals[:5]):
                        product_name_clean = deal['product'].split('.')[1].strip() if '.' in deal['product'] else deal['product']
                        score = deal.get('score', 0)
                        
                        # Color code based on score
                        if score >= 95:
                            color = "🟢"
                        elif score >= 90:
                            color = "🟡"
                        else:
                            color = "🔴"
                        
                        st.markdown(f"**{i+1}. {color} {product_name_clean}**")
                        st.markdown(f"   • Site: `{deal.get('site', 'N/A')}`")
                        st.markdown(f"   • Price: `{deal.get('price', 'N/A')}`")
                        st.markdown(f"   • Deal Score: **:green[{score}/100]**")
                        st.markdown("---")
                else:
                    st.info("No deals found")
            
            # Visualizations Section
            st.markdown("### 📊 ML Analysis Visualizations")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Deal Score Distribution
                if deals and len(deals) >= 3:
                    deal_df = pd.DataFrame(deals)
                    if 'score' in deal_df.columns:
                        fig_scores = px.histogram(
                            deal_df,
                            x='score',
                            nbins=10,
                            title="Deal Score Distribution",
                            labels={'score': 'Deal Score', 'count': 'Number of Products'},
                            color_discrete_sequence=['#00cc66']
                        )
                        fig_scores.update_layout(showlegend=False, height=300)
                        st.plotly_chart(fig_scores, use_container_width=True)
                else:
                    st.info("Not enough data for score distribution")
            
            with viz_col2:
                # Top Deals Bar Chart
                if deals and len(deals) >= 3:
                    top_deals_df = pd.DataFrame(deals[:10])
                    if 'score' in top_deals_df.columns:
                        top_deals_df['product_short'] = top_deals_df['product'].apply(
                            lambda x: x.split('.')[1][:30] + '...' if '.' in x and len(x.split('.')[1]) > 30 else (x.split('.')[1] if '.' in x else x[:30])
                        )
                        fig_bars = px.bar(
                            top_deals_df,
                            x='score',
                            y='product_short',
                            orientation='h',
                            title="Top 10 Deal Scores",
                            labels={'score': 'Deal Score', 'product_short': 'Product'},
                            color='score',
                            color_continuous_scale='Greens'
                        )
                        fig_bars.update_layout(showlegend=False, height=300, yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig_bars, use_container_width=True)
                else:
                    st.info("Not enough data for top deals chart")
            
            # Price Analysis
            if 'price' in df.columns and df['price'].notna().any():
                st.markdown("### 💰 Price Analysis")
                
                price_col1, price_col2 = st.columns(2)
                
                with price_col1:
                    # Price by Site
                    if 'site' in df.columns:
                        fig_site = px.box(
                            df,
                            x='site',
                            y='price',
                            title="Price Distribution by Site",
                            labels={'price': 'Price (PKR)', 'site': 'Store'},
                            color='site'
                        )
                        fig_site.update_layout(showlegend=False, height=300)
                        st.plotly_chart(fig_site, use_container_width=True)
                
                with price_col2:
                    # Unit Price Comparison
                    if 'unit_price' in df.columns and df['unit_price'].notna().any():
                        best_value = df.nsmallest(10, 'unit_price')
                        fig_unit = px.bar(
                            best_value,
                            x='unit_price',
                            y='title',
                            orientation='h',
                            title="Best Value Products (by Unit Price)",
                            labels={'unit_price': 'Unit Price', 'title': 'Product'},
                            color='unit_price',
                            color_continuous_scale='RdYlGn_r'
                        )
                        fig_unit.update_layout(showlegend=False, height=300, yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig_unit, use_container_width=True)
            
            st.success("✅ ML Analysis Complete!")
        
        if models_process.stderr:
            st.error(f"ML Model Error: {models_process.stderr}")
            print(f"\n[ML Models STDERR]\n{models_process.stderr}", file=sys.stderr)

def run_scraper(product_name, max_pages=2):
    """Run productparser.py to scrape data for a product"""
    try:
        # Determine correct path to productparser.py
        if os.path.exists(os.path.join('scrappers', 'productparser.py')):
            scraper_path = os.path.join('scrappers', 'productparser.py')
            working_dir = 'scrappers'
        elif os.path.exists('productparser.py'):
            scraper_path = 'productparser.py'
            working_dir = '.'
        else:
            st.error("❌ productparser.py not found!")
            return None
        
        cmd = [sys.executable, 'productparser.py', product_name, str(max_pages)]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=working_dir
        )
        
        # Stream output
        output_placeholder = st.empty()
        full_output = []
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                full_output.append(output.strip())
                output_placeholder.code('\n'.join(full_output[-20:]))  # Show last 20 lines
        
        process.wait()
        
        # Display any errors to terminal and streamlit
        stderr_output = process.stderr.read()
        if stderr_output:
            print(f"\n[STDERR] {stderr_output}", file=sys.stderr)
        
        if process.returncode != 0:
            st.error(f"Scraper error: {stderr_output}")
            return None
        
        # Now run csvqty.py to process quantities
        csv_file = os.path.join(working_dir, f'scrapper_{product_name}.csv')
        
        if os.path.exists(csv_file):
            st.info("Processing quantities and calculating prices...")
            
            qty_cmd = [sys.executable, 'csvqty.py', f'scrapper_{product_name}.csv']
            qty_process = subprocess.run(
                qty_cmd,
                capture_output=True,
                text=True,
                cwd=working_dir
            )
            
            # Display output and errors to terminal
            if qty_process.stdout:
                print(f"\n[csvqty.py OUTPUT]\n{qty_process.stdout}")
            if qty_process.stderr:
                print(f"\n[csvqty.py STDERR]\n{qty_process.stderr}", file=sys.stderr)
            
            if qty_process.returncode == 0:
                st.success("✅ Quantities processed successfully!")
                qtyfixed_file = os.path.join(working_dir, f'scrapper_{product_name}_qtyfixed.csv')
                
                return qtyfixed_file
            else:
                st.error(f"Quantity processing error: {qty_process.stderr}")
                return None
        
        return None
        
    except Exception as e:
        st.error(f"Error running scraper: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<p class="main-header">🛒 Smart Price Finder</p>', unsafe_allow_html=True)
    st.markdown("**Compare prices across Naheed, Alfatah, Metro, Imtiaz, and Carrefour**")
    
    # Sidebar - Product Search and Scraping
    st.sidebar.header("🔍 Product Search")
    
    # Product search input
    product_search = st.sidebar.text_input(
        "Enter product name to search",
        placeholder="e.g., pepsi, jam, oil"
    )
    
    max_pages = st.sidebar.slider("Pages per store", 1, 5, 2)
    
    if st.sidebar.button("🚀 Search & Scrape", type="primary"):
        if product_search:
            with st.spinner(f"🔍 Searching for '{product_search}' across all stores..."):
                st.info(f"Starting scrape for: **{product_search}**")
                qtyfixed_file = run_scraper(product_search.lower(), max_pages)
                
                if qtyfixed_file and os.path.exists(qtyfixed_file):
                    st.success(f"✅ Successfully scraped and processed data for '{product_search}'!")
                    st.balloons()
                    # Store in session state for immediate display
                    st.session_state['last_scraped_file'] = qtyfixed_file
                    st.session_state['last_scraped_product'] = product_search
                    st.rerun()
                else:
                    st.error("❌ Scraping failed or no data found. Check the output above.")
        else:
            st.warning("⚠️ Please enter a product name to search")
    
    st.sidebar.markdown("---")
    
    # Only show data if we have scraped something
    if 'last_scraped_file' not in st.session_state:
        st.warning("⚠️ No data available yet!")
        st.info("💡 Enter a product name above and click 'Search & Scrape' to get started!")
        return
    
    # Use only the last scraped file
    selected_file = st.session_state['last_scraped_file']
    
    if not os.path.exists(selected_file):
        st.error(f"❌ Scraped file not found: {selected_file}")
        return
    
    # Load data
    df = load_data(selected_file)
    product_name = os.path.basename(selected_file).replace('_analyzed.csv', '').replace('_qtyfixed.csv', '').replace('scrapper_', '')
    
    st.sidebar.success(f"📊 Viewing: {product_name.upper()}")
    
    # Sidebar - Filters
    st.sidebar.header("🔍 Filters")
    
    # Store filter
    all_stores = df['site'].unique().tolist()
    selected_stores = st.sidebar.multiselect(
        "Select Stores",
        all_stores,
        default=all_stores
    )
    
    # Price range filter
    if df['price'].notna().any():
        min_price = float(df['price'].min())
        max_price = float(df['price'].max())
        price_range = st.sidebar.slider(
            "Price Range (PKR)",
            min_price,
            max_price,
            (min_price, max_price)
        )
    else:
        price_range = (0, 1000)
    
    # Size filter
    if 'normalized_quantity' in df.columns and df['normalized_quantity'].notna().any():
        min_size = float(df['normalized_quantity'].min())
        max_size = float(df['normalized_quantity'].max())
        size_range = st.sidebar.slider(
            "Size Range (ml/g)",
            min_size,
            max_size,
            (min_size, max_size)
        )
    else:
        size_range = (0, 10000)
    
    # Apply filters
    filter_mask = (
        (df['site'].isin(selected_stores)) &
        (df['price'] >= price_range[0]) &
        (df['price'] <= price_range[1]) &
        (df['normalized_quantity'] >= size_range[0]) &
        (df['normalized_quantity'] <= size_range[1])
    )
    
    filtered_df = df[filter_mask]
    
    if filtered_df.empty:
        st.warning("⚠️ No products match the selected filters. Try adjusting your criteria.")
        return
    
    # Key Metrics
    st.header("📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cheapest = filtered_df.loc[filtered_df['price'].idxmin()] if filtered_df['price'].notna().any() else None
        if cheapest is not None:
            st.metric(
                "💰 Cheapest Option",
                f"Rs. {cheapest['price']:.0f}",
                f"{cheapest['site']}"
            )
        else:
            st.metric("💰 Cheapest Option", "N/A")
    
    with col2:
        if filtered_df['unit_price'].notna().any():
            best_unit = filtered_df.loc[filtered_df['unit_price'].idxmin()]
            st.metric(
                "⭐ Best Unit Price",
                f"{best_unit['unit_price']:.4f} {best_unit['unit_price_unit']}",
                f"{best_unit['site']}"
            )
        else:
            st.metric("⭐ Best Unit Price", "N/A")
    
    with col3:
        most_available = filtered_df['site'].value_counts().idxmax() if not filtered_df.empty else "N/A"
        count = filtered_df['site'].value_counts().max() if not filtered_df.empty else 0
        st.metric(
            "🏪 Most Options",
            most_available,
            f"{count} products"
        )
    
    with col4:
        st.metric(
            "📦 Total Products",
            len(filtered_df),
            f"{len(all_stores)} stores"
        )
    
    # Show ML Analysis Panel
    run_ml_analysis(selected_file, product_name)
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Quick Compare", "📊 Price Analysis", "💎 Best Deals", "📈 Detailed View"])
    
    with tab1:
        st.subheader(f"🔍 {product_name.upper()} - Price Comparison")
        
        # Show best value products based on unit price
        if filtered_df['unit_price'].notna().any():
            best_deals = filtered_df.nsmallest(5, 'unit_price')[['site', 'title', 'price', 'size_value', 'size_unit', 'unit_price', 'unit_price_unit', 'per_unit_price']]
            st.markdown("### 🏆 Top 5 Best Value Products")
            for idx, row in best_deals.iterrows():
                col_a, col_b, col_c = st.columns([3, 1, 1])
                with col_a:
                    st.markdown(f"**{row['site']}** - {row['title']}")
                with col_b:
                    st.markdown(f"**Rs. {row['price']:.0f}** ({row['size_value']} {row['size_unit']})")
                with col_c:
                    st.markdown(f"🟢 **{row['unit_price']:.4f}** {row['unit_price_unit']}")
                st.markdown("---")
        
        # Side-by-side comparison chart
        if not filtered_df.empty and filtered_df['price'].notna().any():
            fig = px.bar(
                filtered_df.sort_values('price'),
                x='title',
                y='price',
                color='site',
                title=f"Price Comparison - {product_name.upper()}",
                labels={'price': 'Price (PKR)', 'title': 'Product'},
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig, width='stretch')
    
    with tab2:
        st.subheader("📊 Price Analysis & Trends")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Unit price comparison
            if filtered_df['unit_price'].notna().any():
                fig_unit = px.bar(
                    filtered_df.sort_values('unit_price').head(20),
                    x='title',
                    y='unit_price',
                    color='site',
                    title="Unit Price Comparison (PKR/ml or PKR/g)",
                    labels={'unit_price': 'Unit Price', 'title': 'Product'}
                )
                fig_unit.update_layout(xaxis_tickangle=-45, height=400)
                st.plotly_chart(fig_unit, width='stretch')
        
        with col2:
            # Price distribution by store
            if filtered_df['price'].notna().any():
                fig_box = px.box(
                    filtered_df,
                    x='site',
                    y='price',
                    title="Price Distribution by Store",
                    labels={'price': 'Price (PKR)', 'site': 'Store'},
                    color='site'
                )
                fig_box.update_layout(height=400)
                st.plotly_chart(fig_box, width='stretch')
        
        # Scatter plot: Price vs Size
        if filtered_df['price'].notna().any() and 'normalized_quantity' in filtered_df.columns and filtered_df['normalized_quantity'].notna().any():
            # Color by site
            color_col = 'site'
            size_col = 'per_unit_price'
            
            fig_scatter = px.scatter(
                filtered_df,
                x='normalized_quantity',
                y='price',
                color=color_col,
                size=size_col,
                hover_data=['title', 'unit_price', 'pack_count'],
                title="Price vs Size Analysis",
                labels={'normalized_quantity': 'Size (ml/g)', 'price': 'Price (PKR)'}
            )
            fig_scatter.update_layout(height=500)
            st.plotly_chart(fig_scatter, width='stretch')
    
    with tab3:
        st.subheader("💎 Best Deals & Savings")
        
        # Pack savings analysis
        pack_items = filtered_df[filtered_df['pack_count'] > 0].copy()
        if not pack_items.empty:
            st.markdown("### 📦 Pack Deals")
            pack_items['savings_per_pack'] = pack_items.apply(
                lambda row: ((row['price'] / row['pack_count']) if row['pack_count'] > 0 else row['price']), 
                axis=1
            )
            pack_items = pack_items.sort_values('savings_per_pack').head(10)
            
            for idx, row in pack_items.iterrows():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                with col1:
                    st.write(f"**{row['site']}** - {row['title']}")
                with col2:
                    st.write(f"Pack of {int(row['pack_count'])}")
                with col3:
                    st.write(f"Rs. {row['price']:.0f}")
                with col4:
                    st.markdown(f"🟢 **Rs. {row['per_unit_price']:.2f}** /item")
        
        # Store comparison table
        st.markdown("### 🏪 Best Store by Metric")
        store_stats = filtered_df.groupby('site').agg({
            'price': ['mean', 'min', 'count'],
            'unit_price': 'mean'
        }).round(2)
        store_stats.columns = ['Avg Price', 'Min Price', 'Products', 'Avg Unit Price']
        store_stats = store_stats.sort_values('Avg Unit Price')
        st.dataframe(store_stats, width='stretch')
    
    with tab4:
        st.subheader("📈 Detailed Product Table")
        
        # Search/filter
        search_term = st.text_input("🔍 Search products", "")
        if search_term:
            filtered_df = filtered_df[filtered_df['title'].str.contains(search_term, case=False, na=False)]
        
        # Sort options
        sort_options = ['price', 'unit_price', 'per_unit_price', 'normalized_quantity', 'site']
        
        sort_by = st.selectbox(
            "Sort by",
            sort_options
        )
        ascending = st.checkbox("Ascending order", value=True)
        
        display_df = filtered_df.sort_values(sort_by, ascending=ascending)
        
        # Display columns
        display_cols = ['site', 'title', 'price', 'size_value', 'size_unit', 'pack_count', 
                       'unit_price', 'unit_price_unit', 'per_unit_price', 'product_url']
        
        display_cols = [col for col in display_cols if col in display_df.columns]
        
        st.dataframe(
            display_df[display_cols].style.highlight_min(subset=['price', 'unit_price'], color='lightgreen'),
            width='stretch',
            height=600
        )
        
        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name=f"{product_name}_filtered.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"**Data Source:** {selected_file} | **Total Records:** {len(df)} | **Filtered:** {len(filtered_df)}"
    )

if __name__ == "__main__":
    main()
