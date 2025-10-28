#!/usr/bin/env python3
# metro_scraper.py
# Usage:
#   python metro_scraper.py <query> [max_scrolls]
# Example:
#   python metro_scraper.py pepsi 5
#
# This version only keeps products whose title matches the query (token + word-boundary).
#
# Requirements:
#   pip install selenium webdriver-manager

import sys, time, csv, re
from urllib.parse import urljoin, quote_plus
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

BASE = "https://www.metro-online.pk"
HEADLESS = False  # set True to run headless (may need extra handling)

def create_driver(headless=HEADLESS):
    opts = webdriver.ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--window-size=1366,1000")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)

def parse_price(s):
    if not s:
        return None
    s = s.replace(',', '')
    m = re.search(r'(\d+(?:\.\d+)?)', s)
    return float(m.group(1)) if m else None

def title_matches_query_strict(name, query):
    """Require each token in query to appear as a whole word in the product title (case-insensitive)."""
    if not name or not query:
        return False
    # Normalize: remove punctuation and collapse whitespace
    name_lower = re.sub(r'[^\w\s]', ' ', name.lower())
    tokens = [t for t in re.split(r'\s+', query.lower().strip()) if t]
    for t in tokens:
        if not re.search(r'\b' + re.escape(t) + r'\b', name_lower):
            return False
    return True

def scrape_metro(query, max_scrolls=5, out_csv=None, filter_titles=True):
    """Scrape metro search results and only keep titles matching the query tokens (default)."""
    out_csv = out_csv or f"metro_{query}.csv"
    driver = create_driver()
    wait = WebDriverWait(driver, 12)
    try:
        # Metro search uses /search/<query>
        search_url = f"{BASE}/search/{quote_plus(query)}"
        driver.get(search_url)
        print("[i] Opened", search_url)
        print("[i] If the site prompts for region/branch, set it in the opened browser now (if prompted).")
        input("Press Enter after results are visible (or if none required) ...")

        collected = []
        seen = set()

        # Wait initially for product containers
        try:
            wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.CategoryGrid_product_nameAndPricing_div__FwOEW")))
        except Exception:
            pass
        time.sleep(0.6)

        scrolls = 0
        last_count = 0
        while scrolls < max_scrolls:
            # Collect current product info blocks
            blocks = driver.find_elements(By.CSS_SELECTOR, "div.CategoryGrid_product_nameAndPricing_div__FwOEW")
            print(f"[i] Scroll cycle {scrolls+1}: found {len(blocks)} blocks")
            for block in blocks:
                try:
                    # title element
                    try:
                        title_el = block.find_element(By.CSS_SELECTOR, "p.CategoryGrid_product_name__3nYsN")
                        title = title_el.get_attribute("title") or title_el.text.strip()
                    except:
                        title = ""

                    # optional filter: only keep if title contains query tokens
                    if filter_titles:
                        if not title_matches_query_strict(title, query):
                            # skip non-matching titles
                            continue

                    # price element
                    try:
                        price_el = block.find_element(By.CSS_SELECTOR, "p.CategoryGrid_product_price__Svf8T")
                        price = parse_price(price_el.text)
                    except:
                        price = None

                    # product url: find nearest ancestor anchor
                    product_url = ""
                    try:
                        a = block.find_element(By.XPATH, "./ancestor::a[1]")
                        href = a.get_attribute("href")
                        product_url = href if href else ""
                    except:
                        try:
                            parent = block.find_element(By.XPATH, "..")
                            a2 = parent.find_element(By.CSS_SELECTOR, "a")
                            product_url = a2.get_attribute("href") or ""
                        except:
                            product_url = ""

                    # image: look for image within the product card parent
                    img_url = ""
                    try:
                        root = block.find_element(By.XPATH, "ancestor::div[contains(@class,'CategoryGrid_productCard_upper_div__')]")
                        img = root.find_element(By.CSS_SELECTOR, "img")
                        src = img.get_attribute("src") or img.get_attribute("srcset") or ""
                        img_url = src if src.startswith("http") else urljoin(BASE, src) if src else ""
                    except:
                        img_url = ""

                    # product id from URL slug
                    pid = None
                    if product_url:
                        # try to extract trailing numeric id, fallback to last slug
                        m = re.search(r'/([^/]+)/(\d+)$', product_url.rstrip('/'))
                        if m:
                            pid = m.group(2)
                        else:
                            pid = product_url.rstrip('/').split('/')[-1]

                    dedupe_key = product_url or (title.lower() if title else None)
                    if dedupe_key and dedupe_key in seen:
                        continue

                    if dedupe_key:
                        seen.add(dedupe_key)

                    collected.append({
                        "product_id": pid,
                        "title": title,
                        "price": price,
                        "product_url": product_url,
                        "image_url": img_url
                    })
                except Exception:
                    # ignore single-item parse errors
                    continue

            # scroll to bottom to load more items (infinite scroll)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.1)  # wait for new items to load
            scrolls += 1

            # stop if no new items added
            if len(collected) == last_count:
                print("[i] No new items loaded after scroll, stopping.")
                break
            last_count = len(collected)

        # save CSV
        if collected:
            keys = ["product_id", "title", "price", "product_url", "image_url"]
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for r in collected:
                    writer.writerow(r)
            print(f"[+] Saved {len(collected)} rows to {out_csv}")
        else:
            print("[!] No products collected")

    finally:
        driver.quit()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python metro_scraper.py <query> [max_scrolls]\nExample: python metro_scraper.py pepsi 5")
        sys.exit(1)
    term = sys.argv[1]
    sc = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    scrape_metro(term, max_scrolls=sc, out_csv=f"metro_{term}.csv", filter_titles=True)
