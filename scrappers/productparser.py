#!/usr/bin/env python3
"""
Scrapes product data from Pakistani grocery stores

Usage: python productparser.py <product_name> [max_pages]
Example: python productparser.py pepsi 2
"""

import sys
import time
import csv
import os
import re
from urllib.parse import urljoin, quote_plus
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

try:
    from rapidfuzz import fuzz
    FUZZY_AVAILABLE = True
except:
    FUZZY_AVAILABLE = False


def create_driver(headless=False):
    """Setup Chrome browser for scraping"""
    opts = webdriver.ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--window-size=1366,900")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)

def parse_price_from_string(s):
    """Pull out the price number from text"""
    if not s:
        return None
    s = s.replace(',', '')
    m = re.search(r'(\d+(?:\.\d+)?)', s)
    return float(m.group(1)) if m else None

def title_matches_query_strict(name, query):
    """Check if product name actually contains all the search words"""
    if not name or not query:
        return False
    name_lower = re.sub(r'[^\w\s]', ' ', name.lower())
    tokens = [t for t in re.split(r'\s+', query.lower().strip()) if t]
    for t in tokens:
        if not re.search(r'\b' + re.escape(t) + r'\b', name_lower):
            return False
    return True


def scrape_naheed(query, max_pages=2):
    """Get products from Naheed website"""
    BASE = "https://www.naheed.pk"
    results = []
    driver = create_driver(headless=False)
    wait = WebDriverWait(driver, 12)
    
    try:
        search_url = f"{BASE}/catalogsearch/result/?q={quote_plus(query)}"
        driver.get(search_url)
        print(f"[Naheed] Opened {search_url}")
        print("[Naheed] Waiting 7 seconds for page to load...")
        time.sleep(7)
        
        seen = set()
        page = 1
        
        while page <= max_pages:
            try:
                wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "li.item.product.product-item, div.product-item-info")))
            except:
                pass
            time.sleep(0.6)
            
            product_nodes = driver.find_elements(By.CSS_SELECTOR, "li.item.product.product-item")
            if not product_nodes:
                product_nodes = driver.find_elements(By.CSS_SELECTOR, "div.product-item-info")
            print(f"[Naheed] Page {page}: found {len(product_nodes)} products")
            
            for node in product_nodes:
                try:
                    title = ""
                    product_url = ""
                    try:
                        a = node.find_element(By.CSS_SELECTOR, "a.product-item-link, a.product-item-photo")
                        product_url = a.get_attribute("href") or ""
                        # try to get the product name from h2 first
                        try:
                            title_el = node.find_element(By.CSS_SELECTOR, "h2.product.name.product-item-name a.product-item-link")
                            title = title_el.text.strip()
                        except:
                            # otherwise use whatever text we can find
                            title = (a.get_attribute("title") or a.text or "").strip()
                    except:
                        # if that fails, try another way
                        try:
                            title_el = node.find_element(By.CSS_SELECTOR, "h2.product.name.product-item-name a")
                            title = title_el.text.strip()
                            product_url = title_el.get_attribute("href") or product_url
                        except:
                            title = ""
                    
                    price = None
                    try:
                        price_wrapper = node.find_element(By.CSS_SELECTOR, "span.price, span.price-wrapper, .price-box .price")
                        price = parse_price_from_string(price_wrapper.text)
                    except:
                        # sometimes price is in a hidden form field
                        try:
                            hidden_price = node.find_element(By.CSS_SELECTOR, "input[name='price'], input[type='hidden'][name='price']")
                            price = parse_price_from_string(hidden_price.get_attribute("value") or "")
                        except:
                            price = None
                    
                    img_url = ""
                    try:
                        img_el = node.find_element(By.CSS_SELECTOR, "img.product-image-photo, img")
                        src = img_el.get_attribute("src") or img_el.get_attribute("data-src") or ""
                        if src:
                            img_url = src if src.startswith("http") else urljoin(BASE, src)
                    except:
                        img_url = ""
                    
                    dedupe_key = product_url or (title.lower() if title else None)
                    if dedupe_key and dedupe_key in seen:
                        continue
                    
                    # skip products that don't match what we're looking for
                    matched = title_matches_query_strict(title, query)
                    if not matched:
                        continue
                    
                    if dedupe_key:
                        seen.add(dedupe_key)
                    
                    results.append({
                        "site": "Naheed",
                        "title": title,
                        "price": price,
                        "image_url": img_url,
                        "product_url": product_url
                    })
                except:
                    continue
            
            # try to go to next page
            try:
                next_link = None
                try:
                    next_link = driver.find_element(By.CSS_SELECTOR, "a[rel='next']")
                except:
                    try:
                        next_link = driver.find_element(By.XPATH, "//a[contains(., 'Next') or contains(., '›') or contains(@aria-label,'Next')]")
                    except:
                        next_link = None
                
                if not next_link:
                    print("[Naheed] No more pages")
                    break
                
                href = next_link.get_attribute("href")
                if href:
                    print(f"[Naheed] Navigating to next page: {href}")
                    driver.get(href)
                else:
                    next_link.click()
                page += 1
                time.sleep(1.0)
            except:
                break
    finally:
        driver.quit()
    
    print(f"[Naheed] Collected {len(results)} products")
    return results


def scrape_alfatah(query, max_pages=2):
    """Get products from Alfatah website"""
    BASE = "https://alfatah.pk"
    results = []
    driver = create_driver(headless=False)
    wait = WebDriverWait(driver, 12)
    
    try:
        search_url = f"{BASE}/search?q={quote_plus(query)}"
        driver.get(search_url)
        print(f"[Alfatah] Opened {search_url}")
        print("[Alfatah] Waiting 7 seconds for page to load...")
        time.sleep(7)
        
        seen = set()
        page = 1
        
        while page <= max_pages:
            try:
                wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.product-card")))
            except:
                pass
            time.sleep(0.6)
            
            cards = driver.find_elements(By.CSS_SELECTOR, "div.product-card")
            print(f"[Alfatah] Page {page}: found {len(cards)} products")
            
            for card in cards:
                try:
                    name = ""
                    prod_url = ""
                    try:
                        a = card.find_element(By.CSS_SELECTOR, "a.product-title-ellipsis")
                        name = a.text.strip()
                        href = a.get_attribute("href")
                        prod_url = href if href and href.startswith("http") else urljoin(BASE, href) if href else ""
                    except:
                        pass
                    
                    price = None
                    try:
                        p = card.find_element(By.CSS_SELECTOR, "p.product-price")
                        price = parse_price_from_string(p.text)
                    except:
                        pass
                    
                    img = ""
                    try:
                        img_el = card.find_element(By.CSS_SELECTOR, "img")
                        src = img_el.get_attribute("src") or img_el.get_attribute("data-src")
                        if src:
                            img = src if src.startswith("http") else urljoin(BASE, src)
                    except:
                        pass
                    
                    # ignore if product name is empty or doesn't match search
                    if not name or not name.strip():
                        continue
                    if query.lower() not in name.lower():
                        continue
                    
                    dedupe_key = prod_url or name.lower()
                    if dedupe_key in seen:
                        continue
                    
                    seen.add(dedupe_key)
                    results.append({
                        "site": "Alfatah",
                        "title": name,
                        "price": price,
                        "image_url": img,
                        "product_url": prod_url
                    })
                except:
                    continue
            
            # load next page if available
            try:
                next_btn = driver.find_element(By.XPATH, "//a[contains(., 'Next')]")
                href = next_btn.get_attribute("href")
                if href:
                    driver.get(href)
                    page += 1
                    time.sleep(1.0)
                else:
                    break
            except:
                break
    finally:
        driver.quit()
    
    print(f"[Alfatah] Collected {len(results)} products")
    return results


def scrape_metro(query, max_scrolls=5):
    """Get products from Metro - uses scrolling instead of pages"""
    BASE = "https://www.metro-online.pk"
    results = []
    driver = create_driver(headless=False)
    wait = WebDriverWait(driver, 12)
    
    try:
        search_url = f"{BASE}/search/{quote_plus(query)}"
        driver.get(search_url)
        print(f"[Metro] Opened {search_url}")
        print("[Metro] Waiting 7 seconds for page to load...")
        time.sleep(7)
        
        seen = set()
        scrolls = 0
        last_count = 0
        
        while scrolls < max_scrolls:
            try:
                wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.CategoryGrid_product_nameAndPricing_div__FwOEW")))
            except:
                pass
            time.sleep(0.6)
            
            blocks = driver.find_elements(By.CSS_SELECTOR, "div.CategoryGrid_product_nameAndPricing_div__FwOEW")
            print(f"[Metro] Scroll {scrolls+1}: found {len(blocks)} products")
            
            for block in blocks:
                try:
                    title = ""
                    try:
                        title_el = block.find_element(By.CSS_SELECTOR, "p.CategoryGrid_product_name__3nYsN")
                        title = title_el.get_attribute("title") or title_el.text.strip()
                    except:
                        pass
                    
                    # skip if no product name or doesn't match search
                    if not title or not title.strip():
                        continue
                    if query.lower() not in title.lower():
                        continue
                    
                    price = None
                    try:
                        price_el = block.find_element(By.CSS_SELECTOR, "p.CategoryGrid_product_price__Svf8T")
                        price = parse_price_from_string(price_el.text)
                    except:
                        pass
                    
                    product_url = ""
                    try:
                        a = block.find_element(By.XPATH, "./ancestor::a[1]")
                        product_url = a.get_attribute("href") or ""
                    except:
                        pass
                    
                    img_url = ""
                    try:
                        root = block.find_element(By.XPATH, "ancestor::div[contains(@class,'CategoryGrid_productCard_upper_div__')]")
                        img = root.find_element(By.CSS_SELECTOR, "img")
                        src = img.get_attribute("src") or ""
                        img_url = src if src.startswith("http") else urljoin(BASE, src) if src else ""
                    except:
                        pass
                    
                    dedupe_key = product_url or title.lower()
                    if dedupe_key in seen:
                        continue
                    
                    seen.add(dedupe_key)
                    results.append({
                        "site": "Metro",
                        "title": title,
                        "price": price,
                        "image_url": img_url,
                        "product_url": product_url
                    })
                except:
                    continue
            
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.1)
            scrolls += 1
            
            if len(results) == last_count:
                break
            last_count = len(results)
    finally:
        driver.quit()
    
    print(f"[Metro] Collected {len(results)} products")
    return results


def auto_select_location_imtiaz(driver, wait, max_retries=5, retry_delay=3):
    """Automatically pick a location on Imtiaz - they make you select one before showing products"""
    print("[Imtiaz] Waiting for location popup...")
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[Imtiaz] Try #{attempt}/{max_retries} to find location dropdown...")
            time.sleep(retry_delay)
            
            # find the dropdown button and click it
            open_buttons = driver.find_elements(By.CSS_SELECTOR, "button[aria-label='Open']")
            if open_buttons:
                print(f"[Imtiaz] Found {len(open_buttons)} 'Open' buttons!")
                try:
                    btn = open_buttons[-1]
                    driver.execute_script("arguments[0].scrollIntoView(true);", btn)
                    time.sleep(0.3)
                    btn.click()
                    print("[Imtiaz] Clicked area dropdown button.")
                except Exception as e:
                    print(f"[Imtiaz] Couldn't click dropdown: {e}")
                    continue
                
                # wait for the list to show up
                try:
                    list_item_locator = (By.XPATH, "//ul[@role='listbox']//li | //div[contains(@class,'MuiAutocomplete-popper')]//li | //li[contains(@role,'option')]")
                    WebDriverWait(driver, 8).until(EC.presence_of_element_located(list_item_locator))
                    time.sleep(0.4)
                    options = driver.find_elements(*list_item_locator)
                    if options:
                        clicked = False
                        for opt in options:
                            try:
                                if opt.is_displayed():
                                    driver.execute_script("arguments[0].scrollIntoView(true);", opt)
                                    time.sleep(0.15)
                                    opt.click()
                                    clicked = True
                                    print("[Imtiaz] Selected first area option from dropdown.")
                                    break
                            except:
                                continue
                        if not clicked:
                            print("[Imtiaz] Found options but couldn't click any.")
                            continue
                    else:
                        print("[Imtiaz] No options found in dropdown.")
                        continue
                except Exception as e:
                    print(f"[Imtiaz] Timeout or error waiting for area options: {e}")
                    continue
                
                # now hit the Select button to confirm
                try:
                    time.sleep(0.5)
                    select_buttons = driver.find_elements(By.XPATH, "//button[normalize-space()='Select']")
                    clicked_select = False
                    for sb in select_buttons:
                        try:
                            disabled = sb.get_attribute("disabled")
                            classes = (sb.get_attribute("class") or "")
                            if disabled is None and "Mui-disabled" not in classes and "disabled" not in classes:
                                driver.execute_script("arguments[0].scrollIntoView(true);", sb)
                                time.sleep(0.15)
                                sb.click()
                                clicked_select = True
                                print("[Imtiaz] Clicked 'Select' button to confirm location.")
                                break
                        except:
                            continue
                    if not clicked_select:
                        print("[Imtiaz] No enabled 'Select' button found.")
                        continue
                    
                    # done! give it some time to load products
                    wait_seconds = 7
                    print(f"[Imtiaz] Location picked! Waiting {wait_seconds}s for products...")
                    time.sleep(wait_seconds)
                    return
                    
                except Exception as e:
                    print(f"[Imtiaz] Error while trying to click Select button: {e}")
                    continue
            else:
                print(f"[Imtiaz] No 'Open' dropdown buttons found on attempt {attempt}. Retrying...")
                
        except Exception as e:
            print(f"[Imtiaz] Error on attempt {attempt}: {e}")
    
    # if we got here, nothing worked - just continue anyway
    print(f"[Imtiaz] Couldn't pick location after {max_retries} tries. Might not load products...")
    time.sleep(3)

def scrape_imtiaz(query, max_pages=2):
    """Get products from Imtiaz online shop"""
    BASE = "https://shop.imtiaz.com.pk"
    results = []
    driver = create_driver(headless=False)
    wait = WebDriverWait(driver, 15)
    
    try:
        url = f"{BASE}/search?q={query}"
        driver.get(url)
        print(f"[Imtiaz] Opened {url}")
        
        # handle their annoying location picker
        auto_select_location_imtiaz(driver, wait)
        
        seen_ids = set()
        page_no = 1
        
        while page_no <= max_pages:
            product_selector = "div[id^='product-item-']"
            try:
                wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, product_selector)))
            except:
                print("[Imtiaz] Timeout waiting for product cards - continuing anyway")
            time.sleep(0.8)
            
            cards = driver.find_elements(By.CSS_SELECTOR, product_selector)
            print(f"[Imtiaz] Found {len(cards)} cards on page {page_no}")
            
            for card in cards:
                try:
                    # grab product ID
                    pid_attr = card.get_attribute("id") or ""
                    pid = None
                    if pid_attr.startswith("product-item-"):
                        pid = pid_attr.replace("product-item-", "").strip()
                    
                    # get product name
                    try:
                        name_el = card.find_element(By.CSS_SELECTOR, "h4.hazle-product-item_product_item_title__wK9IT")
                    except:
                        try:
                            name_el = card.find_element(By.CSS_SELECTOR, "h4[title]")
                        except:
                            name_el = None
                    name = name_el.get_attribute("title") if name_el else (name_el.text if name_el else "")
                    
                    price = None
                    try:
                        price_span = card.find_element(By.CSS_SELECTOR, "div.hazle-product-item_product_item_price_label__ET_we span")
                        price_text = price_span.text.strip()
                        price = parse_price_from_string(price_text)
                    except:
                        # sometimes it's in a different element
                        try:
                            maybe = card.find_element(By.XPATH, ".//span[contains(., 'Rs') or contains(., 'Rs.') or contains(., 'PKR')]")
                            price = parse_price_from_string(maybe.text)
                        except:
                            price = None
                    
                    img_url = ""
                    try:
                        img_el = card.find_element(By.CSS_SELECTOR, "img")
                        src = img_el.get_attribute("src") or img_el.get_attribute("data-src")
                        if src:
                            img_url = urljoin(BASE, src) if src.startswith("/") else src
                    except:
                        img_url = ""
                    
                    # find product page link
                    product_url = ""
                    try:
                        a = card.find_element(By.CSS_SELECTOR, "a")
                        href = a.get_attribute("href")
                        if href:
                            product_url = href
                    except:
                        product_url = ""
                    
                    # skip duplicates
                    if pid and pid in seen_ids:
                        continue
                    seen_ids.add(pid)
                    
                    results.append({
                        "site": "Imtiaz",
                        "title": name,
                        "price": price,
                        "image_url": img_url,
                        "product_url": product_url
                    })
                except Exception as e:
                    print(f"[Imtiaz] error parsing a card: {e}")
                    continue
            
            # click next if it's there and enabled
            try:
                next_btn = driver.find_element(By.XPATH, "//button[normalize-space()='Next']")
                classes = (next_btn.get_attribute("class") or "")
                disabled_attr = next_btn.get_attribute("disabled")
                disabled = disabled_attr is not None or "Mui-disabled" in classes or "disabled" in classes
                if disabled:
                    print("[Imtiaz] Next button is disabled - we're at the end")
                    break
                
                print("[Imtiaz] Going to next page...")
                # remember first item to see when page actually changes
                first_before = None
                try:
                    first_before = driver.find_element(By.CSS_SELECTOR, product_selector).get_attribute("id")
                except:
                    first_before = None
                next_btn.click()
                
                # wait for page to actually reload
                def first_card_changed(drv):
                    try:
                        first_after = drv.find_element(By.CSS_SELECTOR, product_selector).get_attribute("id")
                        return first_after != first_before
                    except:
                        return True
                WebDriverWait(driver, 15).until(first_card_changed)
                time.sleep(0.6)
                page_no += 1
            except Exception as e:
                print(f"[Imtiaz] Can't find next button or no more pages: {e}")
                break
    finally:
        driver.quit()
    
    print(f"[Imtiaz] Collected {len(results)} products")
    return results


def scrape_carrefour(query, max_pages=2):
    """Get products from Carrefour"""
    BASE = "https://www.carrefour.pk"
    SEARCH_BASE = BASE + "/mafpak/en/v4/search?keyword="
    results = []
    driver = create_driver(headless=False)
    wait = WebDriverWait(driver, 15)
    
    try:
        url = SEARCH_BASE + query
        driver.get(url)
        print(f"[Carrefour] Opened {url}")
        print("[Carrefour] Waiting 7 seconds for page to load...")
        time.sleep(7)
        
        page = 1
        while page <= max_pages:
            product_selector = "ul.css-1omnv59"
            try:
                wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, product_selector)))
            except:
                pass
            time.sleep(0.6)
            
            cards = driver.find_elements(By.CSS_SELECTOR, product_selector)
            print(f"[Carrefour] Page {page}: found {len(cards)} products")
            
            for card in cards:
                try:
                    name = ""
                    product_url = ""
                    try:
                        name_el = card.find_element(By.CSS_SELECTOR, "a[data-testid='product_name']")
                        name = name_el.text.strip()
                        href = name_el.get_attribute("href")
                        product_url = urljoin(BASE, href) if href else ""
                    except:
                        pass
                    
                    image_url = ""
                    try:
                        img = card.find_element(By.CSS_SELECTOR, "img[data-testid='product_image_main']")
                        src = img.get_attribute("src")
                        image_url = src if src.startswith("http") else urljoin(BASE, src)
                    except:
                        pass
                    
                    price = None
                    try:
                        # carrefour splits price into integer and decimal parts
                        int_el = card.find_element(By.CSS_SELECTOR, "div.css-14zpref")
                        frac_el = card.find_element(By.CSS_SELECTOR, "div.css-1pjcwg4")
                        int_text = int_el.text.strip()
                        frac_text = frac_el.text.strip()
                        # combine them
                        int_part = re.sub(r'[^\d]', '', int_text)
                        frac_part = re.search(r'(\d+)', frac_text)
                        frac_part = frac_part.group(1) if frac_part else "00"
                        price = float(f"{int_part}.{frac_part}") if int_part else None
                    except:
                        pass
                    
                    results.append({
                        "site": "Carrefour",
                        "title": name,
                        "price": price,
                        "image_url": image_url,
                        "product_url": product_url
                    })
                except:
                    continue
            
            # check for next page button
            try:
                next_el = driver.find_element(By.CSS_SELECTOR, "a[rel='next']")
                if next_el.get_attribute("disabled"):
                    break
                driver.execute_script("arguments[0].scrollIntoView(true);", next_el)
                time.sleep(0.2)
                next_el.click()
                time.sleep(1.0)
                page += 1
            except:
                break
    finally:
        driver.quit()
    
    print(f"[Carrefour] Collected {len(results)} products")
    return results


def scrape_all_sites(product_name, max_pages=2):
    """
    Hit all 5 stores one by one to scrape product data.
    Returns everything in one big list.
    """
    all_results = []
    
    print(f"\n{'='*60}")
    print(f"Searching for: {product_name}")
    print(f"Pages per store: {max_pages}")
    print(f"{'='*60}\n")
    
    # Go through each store
    print("\n[1/5] Naheed...")
    try:
        results = scrape_naheed(product_name, max_pages)
        all_results.extend(results)
    except Exception as e:
        print(f"[Naheed] Error: {e}")
    
    print("\n[2/5] Alfatah...")
    try:
        results = scrape_alfatah(product_name, max_pages)
        all_results.extend(results)
    except Exception as e:
        print(f"[Alfatah] Error: {e}")
    
    print("\n[3/5] Metro...")
    try:
        results = scrape_metro(product_name, max_scrolls=max_pages*2)
        all_results.extend(results)
    except Exception as e:
        print(f"[Metro] Error: {e}")
    
    print("\n[4/5] Imtiaz...")
    try:
        results = scrape_imtiaz(product_name, max_pages)
        all_results.extend(results)
    except Exception as e:
        print(f"[Imtiaz] Error: {e}")
    
    print("\n[5/5] Carrefour...")
    try:
        results = scrape_carrefour(product_name, max_pages)
        all_results.extend(results)
    except Exception as e:
        print(f"[Carrefour] Error: {e}")
    
    print(f"\n{'='*60}")
    print(f"Total products collected: {len(all_results)}")
    print(f"{'='*60}\n")
    
    return all_results

def save_combined_csv(results, product_name):
    """Save everything to a CSV file - will overwrite if file exists"""
    if not results:
        print("[!] Nothing to save")
        return
    
    filename = f"scrapper_{product_name}.csv"
    keys = ["site", "title", "price", "image_url", "product_url"]
    
    if os.path.exists(filename):
        print(f"[i] Overwriting {filename}")
    
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    
    print(f"[+] Saved {len(results)} items to {filename}")
    
    # show breakdown by store
    from collections import Counter
    site_counts = Counter(r["site"] for r in results)
    print("\nBreakdown:")
    for site, count in site_counts.items():
        print(f"  {site}: {count} products")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python productparser.py <product_name> [max_pages]")
        print("Example: python productparser.py pepsi 2")
        sys.exit(1)
    
    product = sys.argv[1]
    max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    
    results = scrape_all_sites(product, max_pages)
    save_combined_csv(results, product)
    
    print(f"\nDone! Check scrapper_{product}.csv")
