#!/usr/bin/env python3
"""
imtiaz_category_scraper.py

Scrape Imtiaz categories -> subcategories -> products (one CSV for all categories).

Outputs a single CSV with columns:
  site, category, subcategory, product_id, title, price, image_url, product_url

Usage:
    pip install selenium webdriver-manager pandas
    python imtiaz_category_scraper.py --headless --max-pages 4 --max-scrolls 30 --out imtiaz_all.csv

Notes:
 - The script attempts to auto-select a delivery area (Karachi / first area) if Imtiaz shows a location modal.
 - It clicks the "Categories" button on the homepage to enumerate main categories, visits each,
   collects subcategory links, then scrapes each subcategory (scroll + pagination fallback).
"""

import argparse
import time
import re
import os
from urllib.parse import urljoin, urlparse

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

IMTIAZ_BASE = "https://shop.imtiaz.com.pk"


def create_driver(headless=False, implicitly_wait=6):
    opts = webdriver.ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--window-size=1366,900")
    # Uncomment to reduce image loading (may speed up, but sometimes breaks image extraction)
    # opts.add_experimental_option("prefs", {"profile.managed_default_content_settings.images": 2})
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
    raw = m.group(1).replace(",", "")
    try:
        return float(raw)
    except:
        return None


def auto_select_location_imtiaz(driver, wait, max_retries=6):
    """
    Try to close / select location modal automatically.
    It opens the page, looks for the 'Select Area / Sub Region' open button and picks the first option.
    This is tolerant: if the modal isn't present it simply returns.
    """
    try:
        time.sleep(0.8)
        # If an input with placeholder "Select City / Region" exists and area dropdown has an Open button:
        # find the popup indicator buttons and click the last one (area dropdown arrow)
        open_buttons = driver.find_elements(By.CSS_SELECTOR, "button[aria-label='Open'], button.MuiAutocomplete-popupIndicator")
        if open_buttons:
            # usually the second 'Open' is the area dropdown (city first is read-only)
            btn = open_buttons[-1]
            try:
                driver.execute_script("arguments[0].scrollIntoView(true);", btn)
                time.sleep(0.2)
                btn.click()
            except:
                try:
                    btn.click()
                except:
                    pass
            # wait for the options list to appear
            time.sleep(0.6)
            # list item selectors for options
            list_items = driver.find_elements(By.XPATH, "//ul[@role='listbox']//li | //div[contains(@class,'MuiAutocomplete-popper')]//li | //li[contains(@role,'option')]")
            if not list_items:
                # maybe dynamic; try again quickly
                time.sleep(0.4)
                list_items = driver.find_elements(By.XPATH, "//ul[@role='listbox']//li | //div[contains(@class,'MuiAutocomplete-popper')]//li | //li[contains(@role,'option')]")
            if list_items:
                # click the first visible option
                clicked = False
                for opt in list_items:
                    try:
                        if opt.is_displayed():
                            driver.execute_script("arguments[0].scrollIntoView(true);", opt)
                            time.sleep(0.12)
                            opt.click()
                            clicked = True
                            break
                    except:
                        continue
                time.sleep(0.5)
            # click the Select button if available and enabled
            selects = driver.find_elements(By.XPATH, "//button[normalize-space()='Select' or normalize-space()='SELECT' or contains(.,'Select')]")
            for sb in selects:
                try:
                    disabled = sb.get_attribute("disabled")
                    classes = (sb.get_attribute("class") or "")
                    if disabled is None and "Mui-disabled" not in classes and "disabled" not in classes:
                        driver.execute_script("arguments[0].scrollIntoView(true);", sb)
                        time.sleep(0.12)
                        sb.click()
                        time.sleep(1.0)
                        return
                except:
                    continue
    except Exception:
        pass
    # if nothing worked, just return (site often still loads products for default)
    return


def click_categories_button_and_get_main_categories(driver, wait):
    """
    Navigate to Imtiaz homepage, extract main categories from the swiper carousel.
    Returns list of (title, href) tuples.
    """
    driver.get(IMTIAZ_BASE)
    time.sleep(1.5)
    # try auto-select location first (some flows show location modal)
    auto_select_location_imtiaz(driver, wait)
    
    # Wait for category carousel to load
    time.sleep(1.0)
    
    # Extract categories from the swiper carousel on homepage
    # Categories are in <a> tags with href="/catalog/..." inside the swiper
    categories = []
    try:
        # Look for category links in the swiper carousel
        # The structure is: div.swiper > div.swiper-wrapper > div.swiper-slide > a[href*="/catalog/"]
        category_links = driver.find_elements(By.CSS_SELECTOR, "div.swiper-slide a[href*='/catalog/'], a[href*='/catalog/']")
        
        for a in category_links:
            try:
                href = a.get_attribute("href") or ""
                # Get title from img title attribute or text content
                title = ""
                try:
                    img = a.find_element(By.TAG_NAME, "img")
                    title = img.get_attribute("title") or img.get_attribute("alt") or ""
                except:
                    pass
                
                if not title:
                    title = a.text.strip()
                
                # Only include if it's a main category link (format: /catalog/category-name-number)
                if href and "/catalog/" in href:
                    full = href if href.startswith("http") else urljoin(IMTIAZ_BASE, href)
                    # Filter out subcategory links (they have longer paths)
                    path = urlparse(full).path
                    if path.count("/") == 2:  # Only /catalog/something
                        categories.append((title or full.rstrip("/").split("/")[-1], full))
            except Exception as e:
                continue
    except Exception as e:
        print(f"[Imtiaz] Error extracting categories: {e}")
    
    print(f"[Imtiaz] Found {len(categories)} main categories on homepage")
    
    # dedupe preserving order
    seen = set()
    uniq = []
    for t, h in categories:
        if h not in seen:
            seen.add(h)
            if not t:
                t = h.rstrip("/").split("/")[-1].replace("-", " ").title()
            uniq.append((t.strip(), h.strip()))
    
    return uniq


def get_subcategory_links_from_category_page(driver, wait, category_url):
    """
    Given a main category page, return subcategory links found in the grid.
    The HTML for subcategory tiles uses <a href="/catalog/..."> around images/labels in MuiBox containers.
    """
    driver.get(category_url)
    time.sleep(0.8)
    # ensure JS renders
    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.MuiBox-root a[href*='/catalog/'], a[href*='/catalog/']")), timeout=6)
    except:
        time.sleep(0.6)

    links = []
    try:
        # Look for subcategory links - they're in <a> tags with images and titles
        # Structure: <a href="/catalog/..."><div class="MuiBox-root..."><img title="..." alt="..."></a>
        anchors = driver.find_elements(By.CSS_SELECTOR, "a[href*='/catalog/']")
        for a in anchors:
            try:
                href = a.get_attribute("href") or ""
                # Get title from img element inside the anchor
                title = ""
                try:
                    img = a.find_element(By.TAG_NAME, "img")
                    title = img.get_attribute("title") or img.get_attribute("alt") or ""
                except:
                    pass
                
                if not title:
                    # Fallback to text content
                    title = a.text.strip()
                
                if href and "/catalog/" in href:
                    full = href if href.startswith("http") else urljoin(IMTIAZ_BASE, href)
                    # Only add if it's different from the category page itself
                    if full != category_url:
                        links.append((title or full.rstrip("/").split("/")[-1], full))
            except:
                continue
    except:
        pass

    # dedupe & return
    seen = set()
    uniq = []
    for t, h in links:
        if h not in seen:
            seen.add(h)
            if not t:
                t = h.rstrip("/").split("/")[-1].replace("-", " ").title()
            uniq.append((t.strip(), h.strip()))
    return uniq


def progressive_scroll(driver, pause=0.8, max_rounds=30):
    """
    Scroll to bottom repeatedly to lazy-load products.
    Returns after repeated stable rounds.
    """
    last_count = -1
    stable = 0
    rounds = 0
    while rounds < max_rounds and stable < 3:
        rounds += 1
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause)
        # small pause then check count of product cards
        cards = driver.find_elements(By.CSS_SELECTOR, "div[id^='product-item-'], div.hazle-product-item_product_item__FSm1N")
        cur = len(cards)
        if cur == last_count:
            stable += 1
        else:
            stable = 0
            last_count = cur
    time.sleep(0.6)
    return driver.find_elements(By.CSS_SELECTOR, "div[id^='product-item-'], div.hazle-product-item_product_item__FSm1N")


def extract_product_from_card(card):
    """
    Given a Selenium element for an Imtiaz product card, extract fields.
    """
    product_id = ""
    title = ""
    price = None
    image_url = ""
    product_url = ""

    try:
        pid_attr = card.get_attribute("id") or ""
        if pid_attr.startswith("product-item-"):
            product_id = pid_attr.replace("product-item-", "").strip()
    except:
        pass

    # title
    try:
        # common h4 with class
        try:
            h4 = card.find_element(By.CSS_SELECTOR, "h4.hazle-product-item_product_item_title__wK9IT, h4[title]")
            title = (h4.get_attribute("title") or h4.text or "").strip()
        except:
            # fallback: first h4/p text
            try:
                h4 = card.find_element(By.CSS_SELECTOR, "h4, p")
                title = (h4.get_attribute("title") or h4.text or "").strip()
            except:
                pass
    except:
        pass

    # price
    try:
        try:
            price_span = card.find_element(By.CSS_SELECTOR, "div.hazle-product-item_product_item_price_label__ET_we span")
            price_text = (price_span.text or "").strip()
            price = parse_price_from_string(price_text)
        except:
            # fallback: any element inside card containing 'Rs' or 'PKR'
            try:
                maybe = card.find_element(By.XPATH, ".//*[contains(text(),'Rs') or contains(text(),'PKR') or contains(text(),'Rs.')]")
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
            image_url = src if src.startswith("http") else urljoin(IMTIAZ_BASE, src)
    except:
        image_url = ""

    # product_url: first anchor
    try:
        a = card.find_element(By.CSS_SELECTOR, "a")
        href = a.get_attribute("href") or ""
        if href:
            product_url = href if href.startswith("http") else urljoin(IMTIAZ_BASE, href)
    except:
        product_url = ""

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


def find_next_button_and_click(driver):
    """
    Attempt to find a pagination 'Next' button and click it.
    Returns True if clicked & page likely changed, False otherwise.
    """
    try:
        # look for Next text button or pagination button
        candidates = driver.find_elements(By.XPATH, "//button[contains(normalize-space(.),'Next') or contains(normalize-space(.),'›') or contains(.,'Next') or //a[contains(.,'Next')]]")
        # also try anchors
        if not candidates:
            candidates = driver.find_elements(By.XPATH, "//a[contains(normalize-space(.),'Next') or contains(normalize-space(.),'›')]")
        for el in candidates:
            try:
                # skip disabled-looking elements
                disabled = el.get_attribute("disabled")
                classes = (el.get_attribute("class") or "")
                if disabled or "disabled" in classes or "Mui-disabled" in classes:
                    continue
                driver.execute_script("arguments[0].scrollIntoView(true);", el)
                time.sleep(0.12)
                try:
                    el.click()
                except:
                    driver.execute_script("arguments[0].click();", el)
                time.sleep(0.8)
                return True
            except:
                continue
    except:
        pass
    return False


def scrape_subcategory(driver, wait, main_cat_title, subcat_title, subcat_url, max_pages=4, max_scroll_rounds=30):
    """
    Scrape one subcategory (scroll + optional pagination).
    Returns list of product dicts (with category/subcategory)
    """
    results = []
    seen = set()
    page = 1
    current_url = subcat_url

    while page <= max_pages and current_url:
        driver.get(current_url)
        time.sleep(0.6)
        # Some pages may require location selection again; try a gentle auto-select if product count is zero
        try:
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[id^='product-item-'], div.hazle-product-item_product_item__FSm1N")), timeout=6)
        except:
            # attempt progressive scroll anyway
            pass

        cards = progressive_scroll(driver, pause=0.6, max_rounds=max_scroll_rounds)
        # cards already returned as list of elements
        print(f"[Imtiaz] {main_cat_title} -> {subcat_title} page {page} | found {len(cards)} cards")
        for card in cards:
            try:
                info = extract_product_from_card(card)
                title = info.get("title") or ""
                if not title:
                    continue
                dedupe_key = (info.get("product_id") or info.get("product_url") or title.lower())
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                results.append({
                    "site": "Imtiaz",
                    "category": main_cat_title,
                    "subcategory": subcat_title,
                    "product_id": info.get("product_id") or "",
                    "title": title,
                    "price": info.get("price"),
                    "image_url": info.get("image_url"),
                    "product_url": info.get("product_url")
                })
            except:
                continue

        # try pagination Next
        clicked = find_next_button_and_click(driver)
        if not clicked:
            break
        page += 1
        # set current_url to current page for loop; many clicks use JS and not href change, so just continue
        try:
            current_url = driver.current_url
        except:
            current_url = None

    return results


def scrape_imtiaz_all(headless=False, max_pages_per_subcat=3, max_scroll_rounds=30, out_csv="imtiaz_all_products.csv"):
    # Use single driver instance for entire scraping session
    driver = create_driver(headless=headless)
    wait = WebDriverWait(driver, 12)

    all_results = []
    seen_global = set()

    try:
        # Get main categories using the same driver
        main_categories = click_categories_button_and_get_main_categories(driver, wait)
        
        if not main_categories:
            print("[!] No categories found on homepage")
            return all_results

        # iterate main categories with the same driver
        for idx, (main_title, main_href) in enumerate(main_categories, start=1):
            print(f"\n=== ({idx}/{len(main_categories)}) Main category: {main_title} ===")
            
            # get subcategories from main category page (reuse driver)
            try:
                subcats = get_subcategory_links_from_category_page(driver, wait, main_href)
            except Exception as e:
                print(f"[Imtiaz] error getting subcategories for {main_title}: {e}")
                subcats = []

            # if no subcategories found, treat main_href as a subcategory itself
            if not subcats:
                subcats = [(main_title, main_href)]

            for sub_idx, (sub_title, sub_href) in enumerate(subcats, start=1):
                print(f"[Imtiaz] Scraping subcategory ({sub_idx}/{len(subcats)}) {sub_title}")
                try:
                    sub_results = scrape_subcategory(driver, wait, main_title, sub_title, sub_href, max_pages=max_pages_per_subcat, max_scroll_rounds=max_scroll_rounds)
                except Exception as e:
                    print(f"[Imtiaz] Error scraping subcategory {sub_title}: {e}")
                    sub_results = []

                for r in sub_results:
                    key = (r.get("site"), r.get("product_id") or r.get("product_url") or r.get("title"))
                    if key in seen_global:
                        continue
                    seen_global.add(key)
                    all_results.append(r)
                
                # Print first 2 rows of this subcategory if products found
                if sub_results:
                    temp_df = pd.DataFrame(sub_results)
                    print(f"\n[Sample] First 2 products from {sub_title}:")
                    print(temp_df[["title", "price"]].head(2).to_string(index=False))
                    print()

                time.sleep(0.4)

    finally:
        # Close driver only at the very end
        driver.quit()

    # save CSV
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
        print("[!] No products found; CSV not written.")

    return all_results

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Imtiaz category -> subcategory -> product scraper")
    parser.add_argument("--headless", action="store_true", help="Run headless Chrome")
    parser.add_argument("--max-pages", type=int, default=3, help="Max pagination pages per subcategory (default=3)")
    parser.add_argument("--max-scrolls", type=int, default=30, help="Max scroll rounds per page (default=30)")
    parser.add_argument("--out", default="imtiaz_all_products.csv", help="Output CSV filename")
    args = parser.parse_args()

    scrape_imtiaz_all(headless=args.headless, max_pages_per_subcat=args.max_pages, max_scroll_rounds=args.max_scrolls, out_csv=args.out)
