#!/usr/bin/env python3
"""
carrefour_scraper.py

Scrapes Carrefour Pakistan (mafpak) category pages using page-index query:
- increments ?currentPage=N to get additional pages
- stops when "No results found" / product-not-found page is encountered
- extracts product_id (from /p/<id>), title, price, image_url, product_url
- dedupes and saves all products in one CSV

Usage:
    pip install selenium webdriver-manager pandas
    python carrefour_scraper.py --headless --out carrefour_products.csv

Example:
    python carrefour_scraper.py --headless --max-pages 200 --start-page 1 \
        --categories "https://www.carrefour.pk/mafpak/en/n/c/clp_FPAK6000000" \
                   "https://www.carrefour.pk/mafpak/en/n/c/clp_FPAK1700000" \
                   "https://www.carrefour.pk/mafpak/en/n/c/clp_FPAK1500000"
"""

import argparse
import time
import re
from urllib.parse import urljoin, urlparse, urlencode, parse_qs
import pandas as pd

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

SITE_NAME = "Carrefour"
BASE = "https://www.carrefour.pk"


def create_driver(headless=False, implicitly_wait=6):
    opts = webdriver.ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--window-size=1366,900")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
    driver.implicitly_wait(implicitly_wait)
    return driver


def parse_price_from_string(s):
    """Return float or None from strings like '1,800', 'PKR 1,800.00', '1,800.00'."""
    if not s:
        return None
    s = str(s)
    # find number group with possible commas and decimals
    m = re.search(r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)', s.replace('\xa0', ' '))
    if not m:
        return None
    raw = m.group(1).replace(',', '')
    try:
        return float(raw)
    except:
        return None


def make_page_url(base_url, page):
    """Append or replace currentPage param."""
    if "currentPage=" in base_url:
        # naive replace
        base, _, rest = base_url.partition('?')
        q = parse_qs(urlparse(base_url).query, keep_blank_values=True)
        q['currentPage'] = [str(page)]
        return base + "?" + urlencode(q, doseq=True)
    else:
        sep = "&" if "?" in base_url else "?"
        return f"{base_url}{sep}currentPage={page}"


def extract_product_from_card(card, base_url=BASE):
    """
    Given a Selenium element for a product card (the full relative div), extract:
    - title
    - product_url
    - image_url
    - price
    - product_id (from /p/<id> in product_url)
    
    Product card structure:
    <div class="relative w-[134px] max-w-[134px] sm:w-[215px] sm:max-w-[215px] flex justify-between">
        <div class="relative">
            <div class="relative flex overflow-hidden rounded-xl...">
                <a href="/mafpak/en/.../p/295592?..."><img src="..."></a>
                <button>...</button>
            </div>
            <div class="max-w-[134px] sm:max-w-[215px]">
                <a href="..."><div class="text-sm..."><span>Sufi Nuggets Poly Bag 1 kg</span></div></a>
                <div class="flex items-center my-xs">
                    <div class="mr-2xs flex w-fit items-center force-ltr text-c4blue-800">
                        <div class="text-lg leading-5 font-bold md:text-xl">750</div>
                        <div class="ml-px flex flex-col">
                            <div class="text-2xs font-bold leading-[10px]">.00</div>
                            <div class="text-2xs font-medium leading-[10px]">PKR</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """
    title = ""
    product_url = ""
    image_url = ""
    price = None
    product_id = ""

    try:
        # Product link (href with /p/)
        try:
            a = card.find_element(By.CSS_SELECTOR, "a[href*='/p/']")
            href = a.get_attribute("href") or ""
            if href:
                product_url = href if href.startswith("http") else urljoin(base_url, href)
        except Exception:
            product_url = ""

        # title: often inside div span within anchor or a text block
        try:
            # try exact known container:
            title_el = card.find_element(By.CSS_SELECTOR,
                                          "div.text-sm.leading-4.font-medium.line-clamp-2, div.text-sm.leading-4.font-medium, h3, h4, a > div")
            title = (title_el.get_attribute("title") or title_el.text or "").strip()
        except Exception:
            try:
                # fallback to anchor text
                a2 = card.find_element(By.CSS_SELECTOR, "a")
                title = (a2.get_attribute("title") or a2.text or "").strip()
            except Exception:
                title = (card.text or "").splitlines()[0].strip()

        # image
        try:
            img = card.find_element(By.CSS_SELECTOR, "img")
            src = img.get_attribute("src") or img.get_attribute("data-src") or ""
            if src:
                image_url = src if src.startswith("http") else urljoin(base_url, src)
        except Exception:
            image_url = ""

        # price: try structured numeric parts shown in markup first, else fallback to regex
        try:
            # integer part selector used in example
            int_el = None
            try:
                int_el = card.find_element(By.CSS_SELECTOR, "div.text-lg.font-bold, div.text-lg.leading-5.font-bold, .text-lg")
            except:
                int_el = None

            if int_el:
                int_text = int_el.text or ""
                # find decimal sibling if present
                dec_text = ""
                try:
                    # example puts decimals in a nearby small div
                    dec_el = card.find_element(By.CSS_SELECTOR, "div.text-2xs, .text-2xs")
                    dec_text = dec_el.text or ""
                except:
                    dec_text = ""
                combined = f"{int_text}{dec_text}"
                price = parse_price_from_string(combined)
            if price is None:
                # fallback to scanning visible text for PKR/Rs. patterns
                txt = card.text or ""
                price = parse_price_from_string(txt)
            # final fallback: search innerHTML for data-price etc
            if price is None:
                inner = card.get_attribute("innerHTML") or ""
                m = re.search(r'data-price(?:-amount)?\s*=\s*[\'"]?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)', inner)
                if m:
                    price = parse_price_from_string(m.group(1))
        except Exception:
            price = None

        # product id from URL /p/<id>
        if product_url:
            m = re.search(r'/p/(\d+)', product_url)
            if m:
                product_id = m.group(1)
    except Exception:
        pass

    # normalize title
    if isinstance(title, str):
        title = " ".join(title.split())

    return {
        "product_id": product_id,
        "title": title,
        "price": price,
        "image_url": image_url,
        "product_url": product_url
    }


def page_is_no_results(driver):
    """
    Detect 'No results found' page by searching for text nodes or common markers.
    Returns True if it's the product-not-found page.
    """
    try:
        # common visible text
        els = driver.find_elements(By.XPATH, "//*[contains(translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'no results found') "
                                            "or contains(., 'ProductNotFound') or contains(., 'No results found for') or contains(., 'No results found')]")
        if els:
            # ensure it's visible and not just part of other text
            for e in els:
                try:
                    if e.is_displayed() and len((e.text or "").strip()) > 0:
                        return True
                except:
                    continue
    except:
        pass
    return False


def get_subcategories(driver, wait, category_url):
    """
    Extract subcategories from category page.
    Subcategories are displayed as circular images with titles.
    Returns list of (subcategory_name, subcategory_url) tuples
    """
    print(f"[Carrefour] Navigating to category: {category_url}")
    driver.get(category_url)
    time.sleep(1.5)  # Wait for page to load
    
    subcategories = []
    
    try:
        # Find all anchor tags that contain the circular subcategory divs
        # They have rounded-full images and are not product links
        links = driver.find_elements(By.CSS_SELECTOR, "a[href]")
        
        for link in links:
            href = link.get_attribute("href") or ""
            if not href or '/p/' in href:  # Skip product links
                continue
            
            # Check if this link has a circular image (subcategory indicator)
            try:
                img = link.find_element(By.CSS_SELECTOR, "img[class*='rounded-full']")
                subcat_name = img.get_attribute("alt") or ""
                
                if subcat_name and href:
                    full_url = href if href.startswith("http") else urljoin(BASE, href)
                    subcategories.append((subcat_name, full_url))
            except:
                continue
        
        # Deduplicate
        seen_urls = set()
        unique_subcats = []
        for name, url in subcategories:
            if url not in seen_urls:
                seen_urls.add(url)
                unique_subcats.append((name, url))
        
        return unique_subcats
    
    except Exception as e:
        print(f"[Carrefour] Error extracting subcategories: {e}")
        return []


def scrape_subcategory(driver, wait, category_name, subcat_name, subcat_url, max_pages=50):
    """
    Scrape a single subcategory by paginating through currentPage=0,1,2...
    Returns list of product dicts.
    """
    results = []
    seen = set()
    page = 0  # Carrefour starts at 0
    
    while page < max_pages:
        url = make_page_url(subcat_url, page)
        print(f"[{category_name} - {subcat_name}] Page {page} -> {url}")
        driver.get(url)
        
        time.sleep(2.5)  # Wait longer for products to load
        
        # Check for "No results found" BEFORE looking for products
        if page_is_no_results(driver):
            print(f"[{category_name} - {subcat_name}] Page {page} - No results found, moving to next subcategory")
            break
        
        # Find product cards - try multiple selectors
        cards = []
        
        # Try selector 1: max-w divs
        cards = driver.find_elements(By.CSS_SELECTOR, 
            "div.max-w-\\[134px\\], div.max-w-\\[215px\\], div[class*='max-w-[134px]'], div[class*='max-w-[215px]']")
        
        if not cards:
            # Try selector 2: relative divs with product links
            cards = driver.find_elements(By.CSS_SELECTOR, "div.relative:has(a[href*='/p/'])")
        
        if not cards:
            # Try selector 3: any div containing product link
            all_product_links = driver.find_elements(By.CSS_SELECTOR, "a[href*='/p/']")
            card_containers = set()
            for link in all_product_links:
                try:
                    # Find the parent container (usually 2-3 levels up)
                    parent = link.find_element(By.XPATH, "./ancestor::div[contains(@class, 'max-w') or contains(@class, 'relative')][1]")
                    card_containers.add(parent)
                except:
                    continue
            cards = list(card_containers)
        
        print(f"[DEBUG] Found {len(cards)} potential product cards")
        
        # Filter to only those with product links
        product_cards = []
        for card in cards:
            try:
                # Verify it has a product link
                card.find_element(By.CSS_SELECTOR, "a[href*='/p/']")
                product_cards.append(card)
            except:
                continue
        
        print(f"[DEBUG] Filtered to {len(product_cards)} cards with product links")
        cards = product_cards
        
        new_found = 0
        for card in cards:
            try:
                info = extract_product_from_card(card)
                title = info.get("title") or ""
                product_url = info.get("product_url") or ""
                pid = info.get("product_id") or ""
                
                if not title:
                    print(f"[DEBUG] Skipping card - no title extracted")
                    continue
                
                dedupe_key = pid or product_url or title.lower()
                if not dedupe_key or dedupe_key in seen:
                    continue
                
                seen.add(dedupe_key)
                new_found += 1
                results.append({
                    "site": SITE_NAME,
                    "category": category_name,
                    "subcategory": subcat_name,
                    "product_id": pid,
                    "title": title,
                    "price": info.get("price"),
                    "image_url": info.get("image_url"),
                    "product_url": product_url
                })
            except Exception as e:
                print(f"[DEBUG] Error extracting product: {e}")
                continue
        
        print(f"[{category_name} - {subcat_name}] Page {page}: found {len(cards)} cards, {new_found} new products")
        
        # Stop if no new products found
        if new_found == 0:
            if len(cards) == 0:
                print(f"[{category_name} - {subcat_name}] No products on page {page}, stopping")
                break
        
        page += 1
        time.sleep(0.4)  # Throttle
    
    return results


def scrape_category(category_url, driver, wait, max_pages=200, start_page=1, pause=0.9):
    """
    Scrape a single Carrefour category:
    1. Extract subcategories
    2. Print all subcategories
    3. Scrape each subcategory
    4. Print first 2 rows after each subcategory
    
    Returns list of product dicts.
    """
    results = []
    
    # Derive category label from URL
    try:
        parsed = urlparse(category_url)
        segs = [s for s in parsed.path.split("/") if s]
        category_label = segs[-1].replace('-', ' ').title() if segs else category_url
    except:
        category_label = category_url
    
    print(f"\n[Carrefour] === Category: {category_label} ===")
    
    # Extract subcategories
    subcategories = get_subcategories(driver, wait, category_url)
    
    if not subcategories:
        print(f"[Carrefour] No subcategories found for {category_label}")
        return results
    
    # Print all subcategories
    print(f"\n[Carrefour] Found {len(subcategories)} subcategories:")
    print("=" * 60)
    for idx, (name, url) in enumerate(subcategories, start=1):
        print(f"{idx}. {name}")
    print("=" * 60 + "\n")
    
    # Scrape each subcategory
    for idx, (subcat_name, subcat_url) in enumerate(subcategories, start=1):
        print(f"\n[Carrefour] ({idx}/{len(subcategories)}) Scraping subcategory: {subcat_name}")
        
        try:
            subcat_results = scrape_subcategory(driver, wait, category_label, subcat_name, subcat_url, max_pages=max_pages)
            results.extend(subcat_results)
            
            # Print first 2 rows after subcategory finishes
            if subcat_results:
                import pandas as pd
                subcat_df = pd.DataFrame(subcat_results)
                print(f"\n[{category_label} - {subcat_name}] First 2 products:")
                print(subcat_df[['title', 'price']].head(2).to_string(index=False))
                print()
        except Exception as e:
            print(f"[Carrefour] Error scraping {subcat_name}: {e}")
            continue
        
        time.sleep(0.5)  # Pause between subcategories
    
    return results


def scrape_all(start_urls, headless=False, out_csv="carrefour_products.csv", max_pages=200, start_page=1):
    all_results = []
    seen_global = set()
    driver = create_driver(headless=headless, implicitly_wait=6)
    wait = WebDriverWait(driver, 10)
    try:
        for idx, url in enumerate(start_urls, start=1):
            print(f"\n=== ({idx}/{len(start_urls)}) Starting category: {url} ===")
            try:
                cat_results = scrape_category(url, driver, wait, max_pages=max_pages, start_page=start_page)
            except Exception as e:
                print(f"[!] Error scraping {url}: {e}")
                cat_results = []

            # dedupe globally and append
            for r in cat_results:
                key = (r.get("site"), r.get("product_id") or r.get("product_url") or r.get("title"))
                if key in seen_global:
                    continue
                seen_global.add(key)
                all_results.append(r)
            # small pause between categories
            time.sleep(0.6)
    finally:
        driver.quit()

    if all_results:
        df = pd.DataFrame(all_results)
        cols = ["site", "category", "subcategory", "product_id", "title", "price", "image_url", "product_url"]
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        df = df[cols]
        df.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"\n[+] Saved {len(df)} products to {out_csv}")
    else:
        print("[!] No products scraped; CSV not written.")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Carrefour (mafpak) category scraper")
    parser.add_argument("--headless", action="store_true", help="Run Chrome headless")
    parser.add_argument("--out", default="carrefour_products.csv", help="Output CSV filename")
    parser.add_argument("--max-pages", type=int, default=200, help="Maximum pages to try per category")
    parser.add_argument("--start-page", type=int, default=1, help="Start page (currentPage value)")
    parser.add_argument("--categories", nargs="+", default=[
        "https://www.carrefour.pk/mafpak/en/n/c/clp_FPAK6000000",   # frozen (example)
        "https://www.carrefour.pk/mafpak/en/n/c/clp_FPAK1700000",   # grocery
        "https://www.carrefour.pk/mafpak/en/n/c/clp_FPAK1500000"    # beverages
    ], help="Category URLs to scrape (space separated)")
    args = parser.parse_args()

    scrape_all(start_urls=args.categories, headless=args.headless, out_csv=args.out, max_pages=args.max_pages, start_page=args.start_page)
