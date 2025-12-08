#!/usr/bin/env python3
"""
naheed_category_scraper_v3.py

Improved Naheed groceries scraper (v3):
- Starts from /groceries-pets, collects subcategory links
- For each category: progressive scroll to trigger lazy-load, extracts products,
  follows "Next Page" links, dedupes, and writes a CSV
- More aggressive price detection (many fallbacks)

Usage:
    pip install selenium webdriver-manager pandas
    python naheed_category_scraper_v3.py [--headless] [--max-pages N] [--max-scroll-attempts M] [--out filename]

Example:
    python naheed_category_scraper_v3.py --headless --max-pages 4 --max-scroll-attempts 6 --out naheed_grocery_full.csv
"""

import argparse
import time
import re
from urllib.parse import urljoin, urlparse

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

NAHEED_BASE = "https://www.naheed.pk"
GROCERIES_URL = f"{NAHEED_BASE}/groceries-pets"


def create_driver(headless=False, implicitly_wait=6):
    opts = webdriver.ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--window-size=1366,900")
    # Optional: disable images to speed up (comment/uncomment as desired)
    # prefs = {"profile.managed_default_content_settings.images": 2}
    # opts.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
    driver.implicitly_wait(implicitly_wait)
    return driver


def parse_price_from_string(s):
    """Robust extraction of numeric price from strings like 'Rs. 1,000' or '1000' or 'PKR 160'"""
    if not s:
        return None
    s = str(s)
    # try to find numbers with optional Rs/PKR prefix and thousand separators
    m = re.search(r'(?:(?:Rs\.?|PKR)\s*)?(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)', s, flags=re.IGNORECASE)
    if not m:
        return None
    raw = m.group(1)
    raw = raw.replace(',', '')
    try:
        return float(raw)
    except:
        return None


def get_subcategory_links(driver, wait):
    """From GROCERIES_URL get the list of (title, href)."""
    driver.get(GROCERIES_URL)
    print(f"[Naheed] Opened {GROCERIES_URL}")
    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.sub-category-list")), timeout=6)
    except Exception:
        time.sleep(1.2)

    elems = driver.find_elements(By.CSS_SELECTOR, "div.sub-category-list ul li.sub-category-post a, div.sub-category-list ul li a")
    links = []
    for a in elems:
        try:
            href = a.get_attribute("href")
            title = a.get_attribute("title") or a.text or ""
            if href:
                links.append((title.strip(), href.strip()))
        except Exception:
            continue

    # dedupe while preserving order and fallback to path segment if title empty
    seen = set()
    uniq = []
    for t, h in links:
        if h not in seen:
            seen.add(h)
            if not t:
                t = h.rstrip("/").split("/")[-1].replace("-", " ").title()
            uniq.append((t, h))
    print(f"[Naheed] Found {len(uniq)} subcategory links")
    return uniq


def progressive_scroll(driver, pause=1.0, max_attempts=5):
    """
    Scroll down progressively to trigger lazy loading.
    Returns when repeated attempts detect no new height changes, or after max_attempts.
    """
    last_height = driver.execute_script("return document.body.scrollHeight")
    attempts = 0
    while attempts < max_attempts:
        driver.execute_script("window.scrollBy(0, window.innerHeight);")
        time.sleep(pause)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            attempts += 1
        else:
            attempts = 0
            last_height = new_height
    # final short wait to allow JS to render prices/markup
    time.sleep(0.9)


def extract_product_info_from_node(node):
    """
    Given a Selenium element for a product node, extract product_id, title, price, image_url, product_url.
    This version tries many fallbacks for price detection.
    """
    title = ""
    product_url = ""
    image_url = ""
    price = None
    product_id = None

    # Title & URL: try multiple methods to extract text
    try:
        # Method 1: h2 > a.product-item-link
        try:
            a = node.find_element(By.CSS_SELECTOR, "h2.product.name.product-item-name a.product-item-link")
            # Try multiple ways to get text
            title = (a.get_attribute("textContent") or 
                    a.get_attribute("innerText") or 
                    a.text or 
                    a.get_attribute("title") or "").strip()
            product_url = a.get_attribute("href") or ""
        except Exception:
            # Method 2: any product link anchor
            try:
                for selector in ["a.product-item-link", "a.product-item-photo", "a"]:
                    try:
                        a2 = node.find_element(By.CSS_SELECTOR, selector)
                        title = (a2.get_attribute("textContent") or 
                                a2.get_attribute("innerText") or 
                                a2.text or 
                                a2.get_attribute("title") or "").strip()
                        if title:
                            product_url = product_url or (a2.get_attribute("href") or "")
                            break
                    except:
                        continue
            except Exception:
                pass
    except Exception:
        pass

    # Price: try multiple selectors and fallbacks
    try:
        price_selectors = [
            "span.price",
            "span.price-wrapper .price",
            ".price-wrapper .price",
            "div.price-box span.price",
            "div.price-box .price",
            "div.price-box",
            ".price-container",
            "span[class*='price']",
            ".price"
        ]
        found = False
        for sel in price_selectors:
            try:
                elems = node.find_elements(By.CSS_SELECTOR, sel)
                for e in elems:
                    txt = (e.text or "").strip()
                    p = parse_price_from_string(txt)
                    if p is not None:
                        price = p
                        found = True
                        break
                if found:
                    break
            except Exception:
                continue

        # look for descendants with data-price-amount attribute
        if price is None:
            try:
                price_nodes = node.find_elements(By.XPATH, ".//*[@data-price-amount]")
                for pn in price_nodes:
                    val = pn.get_attribute("data-price-amount") or pn.get_attribute("data-price")
                    p = parse_price_from_string(val)
                    if p is not None:
                        price = p
                        break
            except Exception:
                pass

        # fallback: scan innerHTML for data-price-amount="1234"
        if price is None:
            try:
                inner = node.get_attribute("innerHTML") or ""
                m = re.search(r'data-price-amount\s*=\s*["\']?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)["\']?', inner)
                if m:
                    price = parse_price_from_string(m.group(1))
            except Exception:
                pass

        # fallback: hidden input or form value
        if price is None:
            try:
                hidden_price = node.find_element(By.CSS_SELECTOR, "input[name='price'], input[type='hidden'][name='price']")
                price = parse_price_from_string(hidden_price.get_attribute("value") or "")
            except:
                pass

        # last fallback: search for 'Rs' with number in node text
        if price is None:
            try:
                txt = node.text or ""
                m = re.search(r'(?:Rs\.?|PKR)\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)', txt, flags=re.IGNORECASE)
                if m:
                    price = parse_price_from_string(m.group(0))
            except:
                pass

    except Exception:
        price = None

    # Image extraction
    try:
        img = node.find_element(By.CSS_SELECTOR, "img.product-image-photo, img")
        src = img.get_attribute("src") or img.get_attribute("data-src") or img.get_attribute("data-original") or ""
        if src:
            image_url = src if src.startswith("http") else urljoin(NAHEED_BASE, src)
    except:
        image_url = ""

    # product id (hidden input or data attribute or URL)
    try:
        try:
            pid_input = node.find_element(By.CSS_SELECTOR, "input[name='product']")
            pid_val = pid_input.get_attribute("value") or ""
            if pid_val:
                product_id = pid_val
        except:
            pass

        if not product_id:
            try:
                price_box = node.find_element(By.CSS_SELECTOR, "div.price-box[data-product-id]")
                product_id = price_box.get_attribute("data-product-id") or None
            except:
                pass

        if not product_id and product_url:
            # parse trailing numeric id from url path
            try:
                p = urlparse(product_url)
                segs = [s for s in p.path.split("/") if s]
                for s in reversed(segs):
                    if s.isdigit():
                        product_id = s
                        break
            except Exception:
                pass
    except:
        pass

    return {
        "product_id": product_id,
        "title": title,
        "price": price,
        "image_url": image_url,
        "product_url": product_url
    }


def find_next_page_link(driver):
    """
    Attempt to find a 'Next Page' link. Returns href (absolute) or None.
    """
    try:
        # check specific class used by Naheed
        elements = driver.find_elements(By.CSS_SELECTOR, "a.amscroll-next-page, a[rel='next'], a.next, a.btn.btn-primary")
        for e in elements:
            try:
                txt = (e.text or "").strip().lower()
                if "next" in txt or "next page" in txt or "›" in txt or "»" in txt:
                    href = e.get_attribute("href")
                    if href:
                        return urljoin(NAHEED_BASE, href)
            except:
                continue
    except:
        pass

    try:
        cand = driver.find_elements(By.XPATH, "//a[contains(normalize-space(.), 'Next') or contains(normalize-space(.), 'Next Page') or contains(., '›') or contains(., '»')]")
        for e in cand:
            try:
                href = e.get_attribute("href")
                if href:
                    return urljoin(NAHEED_BASE, href)
            except:
                continue
    except:
        pass

    return None


def scrape_category(driver, wait, category_title, category_url, max_pages=4, max_scroll_attempts=5):
    """
    Scrape a single category page (with progressive scroll + pagination).
    Returns list of dicts with keys: site, category, product_id, title, price, image_url, product_url
    """
    results = []
    seen = set()
    page = 1
    current_url = category_url

    while page <= max_pages and current_url:
        driver.get(current_url)
        print(f"\n[Naheed] Category '{category_title}' - Page {page} -> {current_url}")
        try:
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.product-item-info, li.item.product.product-item")), timeout=6)
        except Exception:
            # nothing obvious — continue and attempt scroll
            pass

        # progressive scroll to load lazy items + give extra time for JS to attach prices
        progressive_scroll(driver, pause=1.0, max_attempts=max_scroll_attempts)

        # collect product nodes (use the product-item-info selector)
        nodes = driver.find_elements(By.CSS_SELECTOR, "div.product-item-info, li.item.product.product-item")
        print(f"[Naheed] Found {len(nodes)} product nodes (raw) on page {page}")

        for node in nodes:
            try:
                info = extract_product_info_from_node(node)
                title = info.get("title") or ""
                product_url = info.get("product_url") or ""
                product_id = info.get("product_id") or ""

                # dedupe key: product_id if present else product_url else title
                dedupe_key = product_id or product_url or title.lower()
                if not dedupe_key:
                    continue
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)

                results.append({
                    "site": "Naheed",
                    "category": category_title,
                    "product_id": product_id,
                    "title": title,
                    "price": info.get("price"),
                    "image_url": info.get("image_url"),
                    "product_url": product_url
                })
            except Exception:
                continue

        # look for next page link (Naheed uses a next anchor with ?p=)
        next_href = find_next_page_link(driver)
        if not next_href:
            print(f"[Naheed] No 'Next' link found on page {page}.")
            break

        # prepare for next loop
        page += 1
        current_url = next_href
        time.sleep(0.9)

    return results


def scrape_all_categories(headless=False, max_pages_per_category=4, max_scroll_attempts=5, out_csv="naheed_groceries_products_full.csv"):
    # fetch category list
    tmp = create_driver(headless=headless)
    wait_tmp = WebDriverWait(tmp, 8)
    try:
        categories = get_subcategory_links(tmp, wait_tmp)
    finally:
        tmp.quit()

    all_results = []
    seen_global = set()

    for idx, (cat_title, cat_href) in enumerate(categories, start=1):
        print(f"\n=== ({idx}/{len(categories)}) Category: {cat_title} ===")
        d = create_driver(headless=headless)
        w = WebDriverWait(d, 12)
        try:
            cat_results = scrape_category(d, w, cat_title, cat_href, max_pages=max_pages_per_category, max_scroll_attempts=max_scroll_attempts)
        except Exception as e:
            print(f"[Naheed] Error scraping category {cat_title}: {e}")
            cat_results = []
        finally:
            d.quit()

        for r in cat_results:
            key = (r.get("site"), r.get("product_id") or r.get("product_url") or r.get("title"))
            if key in seen_global:
                continue
            seen_global.add(key)
            all_results.append(r)

        time.sleep(0.6)

    # save to CSV
    if all_results:
        df = pd.DataFrame(all_results)
        cols = ["site", "category", "product_id", "title", "price", "image_url", "product_url"]
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        df = df[cols]
        df.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"\n[+] Saved {len(df)} products to {out_csv}")
    else:
        print("[!] No products scraped; nothing to save.")

    # summary
    from collections import Counter
    cat_counts = Counter(r["category"] for r in all_results)
    print("\nProducts by category (top):")
    for cat, cnt in cat_counts.most_common(15):
        print(f"  {cat}: {cnt}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Naheed groceries category scraper (v3)")
    parser.add_argument("--headless", action="store_true", help="Run Chrome headless")
    parser.add_argument("--max-pages", type=int, default=4, help="Max pages per category to scrape (default=4)")
    parser.add_argument("--max-scroll-attempts", type=int, default=5, help="How many repeated no-change scroll attempts before stopping (default=5)")
    parser.add_argument("--out", default="naheed_groceries_products_full.csv", help="Output CSV filename")
    args = parser.parse_args()

    scrape_all_categories(headless=args.headless, max_pages_per_category=args.max_pages, max_scroll_attempts=args.max_scroll_attempts, out_csv=args.out)
