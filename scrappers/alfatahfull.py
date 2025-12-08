#!/usr/bin/env python3
"""
alfatah_all_categories_one_csv.py

Scrape Alfatah grocery categories and save ALL products from all categories into ONE CSV.

Usage:
    pip install selenium webdriver-manager pandas
    python alfatah_all_categories_one_csv.py [--headless] [--max-scrolls N] [--pause S] [--out file.csv]

Example:
    python alfatah_all_categories_one_csv.py --headless --max-scrolls 40 --pause 0.8 --out alfatah_all_products.csv
"""

import time
import re
import argparse
import os
from urllib.parse import urljoin, urlparse

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

ALFATAH_BASE = "https://alfatah.pk"
GROCERIES_PAGE = ALFATAH_BASE + "/pages/grocery-foods"

def create_driver(headless=False, implicitly_wait=6):
    opts = webdriver.ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--window-size=1366,900")
    # Optional: uncomment the two lines below to block images and speed scraping (may affect some sites)
    # prefs = {"profile.managed_default_content_settings.images": 2}
    # opts.add_experimental_option("prefs", prefs)
    drv = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
    drv.implicitly_wait(implicitly_wait)
    return drv

def parse_price_from_string(s):
    if not s:
        return None
    s = str(s)
    m = re.search(r'(?:(?:Rs\.?|PKR)\s*)?(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)', s, flags=re.IGNORECASE)
    if not m:
        return None
    raw = m.group(1).replace(',', '')
    try:
        return float(raw)
    except:
        return None

def get_category_links(driver, wait):
    driver.get(GROCERIES_PAGE)
    print(f"[Alfatah] Opened {GROCERIES_PAGE}")
    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.box")), timeout=8)
    except Exception:
        time.sleep(1.2)

    links = []
    try:
        elems = driver.find_elements(By.CSS_SELECTOR, "div.box a")
        for a in elems:
            try:
                href = a.get_attribute("href")
                title = ""
                try:
                    title = a.find_element(By.CSS_SELECTOR, ".department-title").text.strip()
                except:
                    title = (a.get_attribute("title") or a.text or "").strip()
                if href:
                    full = href if href.startswith("http") else urljoin(ALFATAH_BASE, href)
                    links.append((title or full.split("/")[-1], full))
            except Exception:
                continue
    except Exception:
        pass

    # fallback broader search if no boxes found
    if not links:
        try:
            elems2 = driver.find_elements(By.CSS_SELECTOR, "a")
            for a in elems2:
                href = a.get_attribute("href")
                if not href:
                    continue
                # heuristic: category links under /collections or /collections/
                if "/collections/" in href or "/pages/" in href or "collection" in href:
                    title = (a.get_attribute("title") or a.text or "").strip()
                    links.append((title or href.split("/")[-1], href if href.startswith("http") else urljoin(ALFATAH_BASE, href)))
        except Exception:
            pass

    # dedupe and normalize
    seen = set()
    uniq = []
    for t,h in links:
        if h not in seen:
            seen.add(h)
            if not t:
                t = h.rstrip("/").split("/")[-1].replace("-", " ").title()
            uniq.append((t.strip(), h.strip()))
    print(f"[Alfatah] Found {len(uniq)} category links")
    return uniq

def scroll_until_no_new_cards(driver, card_selector="div.product-card.card-border, div.product-card, .product-card",
                               max_scrolls=30, pause=0.8, load_more_selectors=None):
    if load_more_selectors is None:
        load_more_selectors = ["button.load-more", "a.load-more", "button[aria-label*='load']", ".load-more", "a[data-action='show_more']"]

    last_count = 0
    stable_rounds = 0
    rounds = 0

    while rounds < max_scrolls and stable_rounds < 3:
        rounds += 1
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause)

        # try clicking "load more" if present
        for sel in load_more_selectors:
            try:
                btns = driver.find_elements(By.CSS_SELECTOR, sel)
                for b in btns:
                    try:
                        if b.is_displayed() and b.is_enabled():
                            driver.execute_script("arguments[0].scrollIntoView(true);", b)
                            time.sleep(0.2)
                            try:
                                b.click()
                            except:
                                driver.execute_script("arguments[0].click();", b)
                            time.sleep(pause)
                    except:
                        continue
            except:
                continue

        time.sleep(0.3)
        cards = driver.find_elements(By.CSS_SELECTOR, card_selector)
        cur_count = len(cards)
        # print progress for debugging:
        # print(f"  scroll {rounds}: {cur_count} cards")
        if cur_count == last_count:
            stable_rounds += 1
        else:
            stable_rounds = 0
            last_count = cur_count

    time.sleep(0.6)
    return driver.find_elements(By.CSS_SELECTOR, card_selector)

def extract_product_from_card(card):
    title = ""
    product_url = ""
    image_url = ""
    price = None
    product_id = ""

    # title & url
    try:
        a = None
        try:
            a = card.find_element(By.CSS_SELECTOR, "a.product-title-ellipsis, a.product-title-ellipsis")
        except:
            try:
                a = card.find_element(By.CSS_SELECTOR, "a")
            except:
                a = None
        if a:
            title = (a.text or a.get_attribute("title") or "").strip()
            product_url = a.get_attribute("href") or ""
    except:
        pass

    # price
    try:
        try:
            p = card.find_element(By.CSS_SELECTOR, "p.product-price, .product-price, div.product-details p.product-price")
            price = parse_price_from_string(p.text)
        except:
            try:
                maybe = card.find_element(By.XPATH, ".//*[contains(text(),'Rs') or contains(text(),'PKR')]")
                price = parse_price_from_string(maybe.text)
            except:
                price = None
    except:
        price = None

    # image
    try:
        img = card.find_element(By.CSS_SELECTOR, "img")
        src = img.get_attribute("src") or img.get_attribute("data-src") or ""
        if src:
            image_url = src if src.startswith("http") else urljoin(ALFATAH_BASE, src)
    except:
        image_url = ""

    # product id heuristic
    try:
        try:
            qty_div = card.find_element(By.CSS_SELECTOR, "div.product-cart-quantity")
            variant_id = qty_div.get_attribute("variant_id") or qty_div.get_attribute("variantid") or ""
            if variant_id:
                m = re.search(r'(\d{5,})', variant_id)
                if m:
                    product_id = m.group(1)
        except:
            pass
    except:
        pass

    if not product_id and product_url:
        try:
            parsed = urlparse(product_url)
            segs = [s for s in parsed.path.split("/") if s]
            for s in reversed(segs):
                m = re.search(r'(\d{5,})', s)
                if m:
                    product_id = m.group(1)
                    break
        except:
            pass

    title = " ".join(title.split()) if isinstance(title, str) else ""
    return {
        "product_id": product_id or "",
        "title": title or "",
        "price": price,
        "image_url": image_url or "",
        "product_url": product_url or ""
    }

def scrape_category_infinitescroll(driver, wait, cat_title, cat_url, max_scrolls=40, pause=0.8):
    print(f"[Alfatah] Category '{cat_title}' -> {cat_url}")
    driver.get(cat_url)
    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.product-card, .product-card, div.product-details")), timeout=8)
    except Exception:
        time.sleep(1.0)

    cards = scroll_until_no_new_cards(driver, max_scrolls=max_scrolls, pause=pause)
    print(f"[Alfatah] After scrolling: found {len(cards)} raw product cards")

    results = []
    seen = set()
    for card in cards:
        try:
            info = extract_product_from_card(card)
            title = info["title"] or ""
            if not title:
                try:
                    t_el = card.find_element(By.CSS_SELECTOR, "div.product-details .product-title-ellipsis a, .product-title a, .product-title-ellipsis")
                    title = (t_el.text or t_el.get_attribute("title") or "").strip()
                    info["title"] = title
                except:
                    pass
            if not title or not title.strip():
                continue

            dedupe_key = (info.get("product_id") or info.get("product_url") or title.lower())
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            results.append({
                "site": "Alfatah",
                "category": cat_title,
                "product_id": info.get("product_id") or "",
                "title": title,
                "price": info.get("price"),
                "image_url": info.get("image_url"),
                "product_url": info.get("product_url")
            })
        except Exception:
            continue

    return results

def scrape_all(headless=False, max_scrolls_per_cat=40, pause=0.8, out="alfatah_all_products.csv"):
    driver = create_driver(headless=headless)
    wait = WebDriverWait(driver, 10)
    try:
        categories = get_category_links(driver, wait)
    finally:
        driver.quit()

    all_results = []
    seen_global = set()

    for idx, (cat_title, cat_href) in enumerate(categories, start=1):
        print(f"\n=== ({idx}/{len(categories)}) Category: {cat_title} ===")
        d = create_driver(headless=headless)
        w = WebDriverWait(d, 10)
        try:
            cat_results = scrape_category_infinitescroll(d, w, cat_title, cat_href, max_scrolls=max_scrolls_per_cat, pause=pause)
        except Exception as e:
            print(f"[Alfatah] Error scraping {cat_title}: {e}")
            cat_results = []
        finally:
            d.quit()

        for r in cat_results:
            key = (r["site"], r["product_id"] or r["product_url"] or r["title"])
            if key in seen_global:
                continue
            seen_global.add(key)
            all_results.append(r)
        
        # Print first 2 rows of this category if products found
        if cat_results:
            temp_df = pd.DataFrame(cat_results)
            print(f"\n[Sample] First 2 products from {cat_title}:")
            print(temp_df[["title", "price"]].head(2).to_string(index=False))
            print()

        time.sleep(0.4)

    # save single CSV with consistent columns
    if all_results:
        df = pd.DataFrame(all_results)
        cols = ["site", "category", "product_id", "title", "price", "image_url", "product_url"]
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        df = df[cols]
        df.to_csv(out, index=False, encoding="utf-8")
        print(f"[+] Saved {len(df)} products to {out}")
    else:
        print("[!] No products scraped.")

    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alfatah scraper: all categories -> single CSV")
    parser.add_argument("--headless", action="store_true", help="Run Chrome headless")
    parser.add_argument("--max-scrolls", type=int, default=40, help="Max scroll attempts per category (default=40)")
    parser.add_argument("--pause", type=float, default=0.8, help="Pause (seconds) after each scroll (default=0.8)")
    parser.add_argument("--out", default="alfatah_all_products.csv", help="Output CSV file")
    args = parser.parse_args()

    scrape_all(headless=args.headless, max_scrolls_per_cat=args.max_scrolls, pause=args.pause, out=args.out)
