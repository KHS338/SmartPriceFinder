#!/usr/bin/env python3
"""
alfatah_scraper.py
Scrapes Alfatah search results for a given query and saves a filtered CSV containing only items
whose titles match the search term.

Usage:
    python alfatah_scraper.py <query> [max_pages] [fuzzy]

Examples:
    python alfatah_scraper.py pepsi 3
    python alfatah_scraper.py "diet pepsi" 2 fuzzy

Options:
    max_pages  - optional integer, default 3
    fuzzy      - optional string. If present, enables fuzzy matching (requires rapidfuzz)

Requirements:
    pip install selenium webdriver-manager
    (optional) pip install rapidfuzz

Notes:
- The script launches a real browser (headless=False by default) so you can select branch/location
  if the site prompts for it. To run headless, set HEADLESS = True below and ensure branch/region
  is set programmatically.
- Filtering is applied DURING scraping to avoid collecting unrelated suggestion cards.
- There are two matching modes:
    * strict: token + word-boundary matching (default, fast and reliable)
    * fuzzy: fallback that uses RapidFuzz to allow tolerant matches (slower)

Output CSV: alfatah_<query>.csv
Optional skipped CSV (for debugging): alfatah_<query>_skipped.csv
"""

import sys
import time
import csv
import re
from urllib.parse import urljoin, quote_plus
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ------- CONFIG -------
BASE = "https://alfatah.pk"
HEADLESS = False  # set True to run headless (you will need to set branch/region programmatically)

# ------- helpers -------

def create_driver(headless=HEADLESS):
    opts = webdriver.ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--window-size=1366,900")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)


def parse_price_from_string(s):
    if not s:
        return None
    s = s.replace(',', '')
    m = re.search(r'(\d+(?:\.\d+)?)', s)
    if m:
        try:
            return float(m.group(1))
        except:
            return None
    return None


def title_matches_query_strict(name, query):
    """Strict matching: require each token in query to appear as a word in the product title.
    Case-insensitive. Uses word-boundary so tokens like 'pepsi' match 'Pepsi 1.5 lt' but not 'xpepsi'.
    """
    if not name or not query:
        return False
    name_lower = re.sub(r"[^\w\s]", ' ', name.lower())
    query = query.lower().strip()
    tokens = [t for t in re.split(r"\s+", query) if t]
    for t in tokens:
        pattern = r"\b" + re.escape(t) + r"\b"
        if not re.search(pattern, name_lower):
            return False
    return True


# optional fuzzy matcher using rapidfuzz (only if user enabled fuzzy mode)
FUZZY_AVAILABLE = False
try:
    from rapidfuzz import fuzz
    FUZZY_AVAILABLE = True
except Exception:
    FUZZY_AVAILABLE = False


def title_matches_with_fuzzy(name, query, token_threshold=90, fuzzy_threshold=80):
    """Try strict first, then fuzzy heuristics if rapidfuzz is available.
    - token_threshold: partial ratio threshold for any token
    - fuzzy_threshold: token_set_ratio threshold for whole-title vs query
    """
    if title_matches_query_strict(name, query):
        return True
    if not FUZZY_AVAILABLE:
        return False
    # fuzzy on whole title vs query
    try:
        score = fuzz.token_set_ratio(name, query)
        if score >= fuzzy_threshold:
            return True
    except Exception:
        pass
    # fallback: check tokens
    for t in [tk for tk in re.split(r"\s+", query) if tk]:
        try:
            if fuzz.partial_ratio(t, name) >= token_threshold:
                return True
        except Exception:
            pass
    return False


# ------- main scraper -------

def scrape_alfatah(query, max_pages=3, out_csv=None, headless=HEADLESS, use_fuzzy=False, save_skipped=False):
    out_csv = out_csv or f"alfatah_{query}.csv"
    skipped_csv = f"alfatah_{query}_skipped.csv" if save_skipped else None

    driver = create_driver(headless=headless)
    wait = WebDriverWait(driver, 12)
    try:
        search_url = f"{BASE}/search?q={quote_plus(query)}&options%5Bprefix%5D=last"
        driver.get(search_url)
        print("[i] Opened", search_url)
        print("[i] If the site prompts for region/branch, please select it now in the browser window.")
        input("Press Enter here after the results load (or if none required) ...")

        results = []
        skipped = []
        page = 1
        seen = set()

        # choose matching function
        if use_fuzzy and not FUZZY_AVAILABLE:
            print("[!] fuzzy mode requested but 'rapidfuzz' not installed. Falling back to strict mode.")
            use_fuzzy = False

        match_fn = title_matches_with_fuzzy if use_fuzzy else title_matches_query_strict

        while page <= max_pages:
            try:
                wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.product-card, div.col-6")))
            except Exception as e:
                print("[!] Timeout waiting for product cards (continuing):", e)
            time.sleep(0.6)

            # product cards
            cards = driver.find_elements(By.CSS_SELECTOR, "div.product-card")
            if not cards:
                cards = driver.find_elements(By.CSS_SELECTOR, "div.col-6 div.product-card")
            print(f"[i] Page {page}: found {len(cards)} product cards")

            for card in cards:
                try:
                    # title & URL
                    name = ""
                    prod_url = ""
                    try:
                        a = card.find_element(By.CSS_SELECTOR, "a.product-title-ellipsis")
                        name = a.text.strip()
                        href = a.get_attribute("href")
                        prod_url = href if href and href.startswith("http") else urljoin(BASE, href) if href else ""
                    except Exception:
                        try:
                            a2 = card.find_element(By.CSS_SELECTOR, "a")
                            name = a2.text.strip() or a2.get_attribute("title") or ""
                            href = a2.get_attribute("href")
                            prod_url = href if href and href.startswith("http") else urljoin(BASE, href) if href else ""
                        except Exception:
                            name = ""

                    # product id from link slug if present
                    pid = None
                    if prod_url:
                        m = re.search(r'/products/([^/?#]+)', prod_url)
                        if m:
                            pid = m.group(1)

                    # price
                    price = None
                    try:
                        p = card.find_element(By.CSS_SELECTOR, "p.product-price")
                        price = parse_price_from_string(p.text)
                    except Exception:
                        try:
                            maybe = card.find_element(By.XPATH, ".//*[contains(text(),'Rs') or contains(text(),'PKR')]")
                            price = parse_price_from_string(maybe.text)
                        except Exception:
                            price = None

                    # image
                    img = ""
                    try:
                        img_el = card.find_element(By.CSS_SELECTOR, "img")
                        src = img_el.get_attribute("src") or img_el.get_attribute("data-src")
                        if src:
                            img = src if src.startswith("http") else urljoin(BASE, src)
                    except Exception:
                        img = ""

                    dedupe_key = prod_url or (name.lower() if name else None)
                    if dedupe_key and dedupe_key in seen:
                        continue

                    # apply filter DURING scraping
                    matched = match_fn(name, query)
                    if not matched:
                        skipped.append({"product_id": pid, "name": name, "price": price, "product_url": prod_url})
                        continue

                    if dedupe_key:
                        seen.add(dedupe_key)

                    results.append({
                        "product_id": pid,
                        "name": name,
                        "price": price,
                        "image_url": img,
                        "product_url": prod_url,
                        "page": page
                    })
                except Exception as e:
                    # skip problematic cards but don't abort
                    print("[!] error parsing a card (skipping):", e)
                    continue

            # pagination - Alfatah often uses links for next page
            try:
                next_btn = None
                try:
                    next_btn = driver.find_element(By.XPATH, "//a[contains(., 'Next') or contains(., '›') or contains(@rel,'next')]")
                except Exception:
                    try:
                        next_btn = driver.find_element(By.CSS_SELECTOR, "ul.pagination li.next a")
                    except Exception:
                        next_btn = None

                if not next_btn:
                    print("[i] No Next link found — stopping pagination.")
                    break

                href = next_btn.get_attribute("href")
                if href:
                    print("[i] Navigating to next page:", href)
                    driver.get(href)
                else:
                    print("[i] Clicking Next button")
                    next_btn.click()

                page += 1
                time.sleep(1.0)
            except Exception as e:
                print("[i] Pagination ended or error:", e)
                break

        # save CSV
        if results:
            keys = ["product_id", "name", "price", "image_url", "product_url", "page"]
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for r in results:
                    writer.writerow(r)
            print(f"[+] Saved {len(results)} rows to {out_csv}")
        else:
            print("[!] No results collected")

        if save_skipped and skipped:
            keys = ["product_id", "name", "price", "product_url"]
            with open(skipped_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for r in skipped:
                    writer.writerow(r)
            print(f"[i] Saved {len(skipped)} skipped rows to {skipped_csv} for tuning")

    finally:
        driver.quit()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python alfatah_scraper.py <query> [max_pages] [fuzzy] [save_skipped]")
        sys.exit(1)
    term = sys.argv[1]
    pages = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    fuzzy_flag = False
    if len(sys.argv) > 3 and sys.argv[3].lower() in ("fuzzy", "--fuzzy"):
        fuzzy_flag = True
    save_skipped_flag = False
    if len(sys.argv) > 4 and sys.argv[4].lower() in ("save_skipped", "--save-skipped"):
        save_skipped_flag = True

    if fuzzy_flag and not FUZZY_AVAILABLE:
        print("[!] You enabled fuzzy matching but 'rapidfuzz' is not installed. To install: pip install rapidfuzz")

    scrape_alfatah(term, max_pages=pages, out_csv=f"alfatah_{term}.csv", headless=HEADLESS, use_fuzzy=fuzzy_flag, save_skipped=save_skipped_flag)
