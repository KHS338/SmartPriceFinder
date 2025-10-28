#!/usr/bin/env python3
# naheed_scraper.py
# Usage:
#   python naheed_scraper.py pepsi 3
#   python naheed_scraper.py "diet pepsi" 2 fuzzy save_skipped
#
# Requirements:
#   pip install selenium webdriver-manager
#   (optional) pip install rapidfuzz

import sys, time, csv, re
from urllib.parse import urljoin, quote_plus
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

BASE = "https://www.naheed.pk"
HEADLESS = False  # change True for headless runs (may require cookie/region handling)

# optional fuzzy
FUZZY_AVAILABLE = False
try:
    from rapidfuzz import fuzz
    FUZZY_AVAILABLE = True
except Exception:
    FUZZY_AVAILABLE = False

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
    return float(m.group(1)) if m else None

def title_matches_query_strict(name, query):
    """Strict token + word-boundary match (case-insensitive)."""
    if not name or not query:
        return False
    name_lower = re.sub(r'[^\w\s]', ' ', name.lower())
    tokens = [t for t in re.split(r'\s+', query.lower().strip()) if t]
    for t in tokens:
        if not re.search(r'\b' + re.escape(t) + r'\b', name_lower):
            return False
    return True

def title_matches_with_fuzzy(name, query, token_threshold=90, fuzzy_threshold=80):
    """Use strict first; if not matched, and rapidfuzz available, use fuzz tests."""
    if title_matches_query_strict(name, query):
        return True
    if not FUZZY_AVAILABLE:
        return False
    try:
        if fuzz.token_set_ratio(name, query) >= fuzzy_threshold:
            return True
    except Exception:
        pass
    for t in [tk for tk in re.split(r'\s+', query) if tk]:
        try:
            if fuzz.partial_ratio(t, name) >= token_threshold:
                return True
        except Exception:
            pass
    return False

def scrape_naheed(query, max_pages=3, out_csv=None, use_fuzzy=False, save_skipped=False, headless=HEADLESS):
    out_csv = out_csv or f"naheed_{query}.csv"
    skipped_csv = f"naheed_{query}_skipped.csv" if save_skipped else None

    driver = create_driver(headless=headless)
    wait = WebDriverWait(driver, 12)
    try:
        search_url = f"{BASE}/catalogsearch/result/?q={quote_plus(query)}"
        driver.get(search_url)
        print("[i] Opened", search_url)
        print("[i] If Naheed prompts for region/branch or a popup appears, please handle it in the opened browser window.")
        input("Press Enter after search results are visible (or if none required) ...")

        results = []
        skipped = []
        seen = set()
        page = 1

        if use_fuzzy and not FUZZY_AVAILABLE:
            print("[!] fuzzy requested but rapidfuzz not installed — falling back to strict mode.")
            use_fuzzy = False

        match_fn = title_matches_with_fuzzy if use_fuzzy else title_matches_query_strict

        while page <= max_pages:
            # wait for product items
            try:
                wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "li.item.product.product-item, div.product-item-info")))
            except Exception:
                # proceed anyway
                pass
            time.sleep(0.6)

            product_nodes = driver.find_elements(By.CSS_SELECTOR, "li.item.product.product-item")
            if not product_nodes:
                # try alternate selector
                product_nodes = driver.find_elements(By.CSS_SELECTOR, "div.product-item-info")
            print(f"[i] Page {page}: found ~{len(product_nodes)} product nodes")

            for node in product_nodes:
                try:
                    # title & product_url
                    title = ""
                    product_url = ""
                    try:
                        a = node.find_element(By.CSS_SELECTOR, "a.product-item-link, a.product-item-photo, a.product-item-link")
                        product_url = a.get_attribute("href") or ""
                        # prefer h2.product-item-name anchor text if present
                        try:
                            title_el = node.find_element(By.CSS_SELECTOR, "h2.product.name.product-item-name a.product-item-link")
                            title = title_el.text.strip()
                        except Exception:
                            # fallback to anchor text or image alt
                            title = (a.get_attribute("title") or a.text or "").strip()
                    except Exception:
                        # try other title selector
                        try:
                            title_el = node.find_element(By.CSS_SELECTOR, "h2.product.name.product-item-name a")
                            title = title_el.text.strip()
                            product_url = title_el.get_attribute("href") or product_url
                        except Exception:
                            title = ""

                    # price: element with class 'price' inside price wrapper
                    price = None
                    try:
                        price_wrapper = node.find_element(By.CSS_SELECTOR, "span.price, span.price-wrapper, .price-box .price")
                        price = parse_price_from_string(price_wrapper.text)
                    except Exception:
                        # try input hidden price in form
                        try:
                            hidden_price = node.find_element(By.CSS_SELECTOR, "input[name='price'], input[type='hidden'][name='price']")
                            price = parse_price_from_string(hidden_price.get_attribute("value") or "")
                        except Exception:
                            price = None

                    # image
                    img_url = ""
                    try:
                        img_el = node.find_element(By.CSS_SELECTOR, "img.product-image-photo, img")
                        src = img_el.get_attribute("src") or img_el.get_attribute("data-src") or ""
                        if src:
                            img_url = src if src.startswith("http") else urljoin(BASE, src)
                    except Exception:
                        img_url = ""

                    # product id: from product-item-info id or data attributes or URL
                    pid = None
                    try:
                        pid_attr = node.get_attribute("id") or ""
                        m = re.search(r'product-item-info_(\d+)', pid_attr)
                        if m:
                            pid = m.group(1)
                    except Exception:
                        pid = None
                    if not pid and product_url:
                        # fallback: last numeric in URL or last slug
                        m2 = re.search(r'/product/(\d+)', product_url)
                        if m2:
                            pid = m2.group(1)
                        else:
                            pid = product_url.rstrip('/').split('/')[-1]

                    dedupe_key = product_url or (title.lower() if title else None)
                    if dedupe_key and dedupe_key in seen:
                        continue

                    # apply filter during scraping
                    matched = match_fn(title, query)
                    if not matched:
                        skipped.append({"product_id": pid, "title": title, "price": price, "product_url": product_url})
                        continue

                    if dedupe_key:
                        seen.add(dedupe_key)

                    results.append({
                        "product_id": pid,
                        "title": title,
                        "price": price,
                        "image_url": img_url,
                        "product_url": product_url,
                        "page": page
                    })
                except Exception as e:
                    # skip single node errors
                    # print("[!] node parse error:", e)
                    continue

            # pagination: try rel="next" link, then pagination anchors
            try:
                next_link = None
                try:
                    next_link = driver.find_element(By.CSS_SELECTOR, "a[rel='next']")
                except Exception:
                    try:
                        # try pagination with text 'Next' or an arrow
                        next_link = driver.find_element(By.XPATH, "//a[contains(., 'Next') or contains(., '›') or contains(@aria-label,'Next')]")
                    except Exception:
                        next_link = None

                if not next_link:
                    print("[i] No next link — stopping pagination.")
                    break

                href = next_link.get_attribute("href")
                if href:
                    print("[i] Navigating to next page:", href)
                    driver.get(href)
                else:
                    # click fallback
                    next_link.click()
                page += 1
                time.sleep(1.0)
            except Exception as e:
                print("[i] Pagination ended or error:", e)
                break

        # save CSV
        if results:
            keys = ["product_id", "title", "price", "image_url", "product_url", "page"]
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for r in results:
                    writer.writerow(r)
            print(f"[+] Saved {len(results)} rows to {out_csv}")
        else:
            print("[!] No matching products collected (check query/match rules)")

        if save_skipped and skipped:
            keys = ["product_id", "title", "price", "product_url"]
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
        print("Usage: python naheed_scraper.py <query> [max_pages] [fuzzy] [save_skipped]")
        sys.exit(1)
    term = sys.argv[1]
    pages = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    fuzzy_flag = False
    save_skipped_flag = False
    if len(sys.argv) > 3 and sys.argv[3].lower() in ("fuzzy", "--fuzzy"):
        fuzzy_flag = True
    if len(sys.argv) > 4 and sys.argv[4].lower() in ("save_skipped", "--save-skipped"):
        save_skipped_flag = True

    if fuzzy_flag and not FUZZY_AVAILABLE:
        print("[!] fuzzy requested but 'rapidfuzz' is not installed. Install with: pip install rapidfuzz")

    scrape_naheed(term, max_pages=pages, out_csv=f"naheed_{term}.csv", use_fuzzy=fuzzy_flag, save_skipped=save_skipped_flag)
