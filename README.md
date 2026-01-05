# SmartPriceFinder

A comprehensive price comparison and product scraping tool for grocery stores in Pakistan. This project helps users find the best deals by scraping and comparing prices from multiple retailers.

## Features

- **Multi-Store Scraping**: Collects product data from major grocery retailers including:
  - Alfatah
  - Imtiaz
  - Naheed
  - Rainbow
  - Carrefour
  - Metro

- **Price Comparison**: Compares prices across different stores to find the best deals
- **Product Matching**: Uses fuzzy matching to identify similar products across stores
- **Data Analysis**: Provides insights into pricing trends and best value products
- **CSV Export**: Exports product data and comparisons to CSV files

## Project Structure

```
SmartPriceFinder/
├── scrappers/           # Web scraping scripts for each store
│   ├── alfatah.py      # Alfatah store scraper
│   ├── imtiaz.py       # Imtiaz store scraper
│   ├── naheed.py       # Naheed store scraper
│   ├── rainbow.py      # Rainbow store scraper
│   ├── carrefour.py    # Carrefour store scraper
│   ├── metro.py        # Metro store scraper
│   ├── csvfeatures.py  # CSV feature extraction
│   ├── csvqty.py       # Quantity parsing utilities
│   ├── productparser.py # Product data parser
│   └── fuzzy_match_products.py # Product matching algorithm
├── models/             # Machine learning models (not included)
├── fullproductscsvs/   # Complete product datasets
└── misc/               # Additional utilities and scripts
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YourUsername/SmartPriceFinder.git
cd SmartPriceFinder
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Individual Scrapers

```python
# Scrape Alfatah products
python scrappers/alfatahfull.py

# Scrape Naheed products
python scrappers/naheedfull.py

# Scrape Imtiaz products
python scrappers/imtiazfull.py
```

### Price Comparison

The unified product comparison data is available in:
```
fullproductscsvs/unified_products_with_price_comparison.csv
```

## Data Files

The project maintains several CSV files:
- **Product Data**: Complete product listings from each store
- **Feature Extracted Data**: Products with parsed features (quantity, brand, etc.)
- **Unified Data**: Combined data from all stores with price comparisons

## Requirements

- Python 3.8+
- BeautifulSoup4
- Requests
- Pandas
- Selenium (for dynamic content scraping)
- fuzzywuzzy (for product matching)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is for educational and personal use only. Please respect the terms of service of the websites being scraped.

## Disclaimer

This tool is intended for price comparison and educational purposes. Users should comply with the robots.txt and terms of service of all websites being scraped.
