#!/usr/bin/env python3
"""
rainbow_scraper.py

Scrape Rainbow Cash & Carry (rainbowcc.com.pk) groceries:

- Auto-select location popup (tries to choose Lahore / Johar Town if available)
- Collect category links from the carousel
- For each category: iterate "Sections" radio options (each becomes a subcategory)
- **Uses scroll-based loading by default (no "Next Page")** — specify --pagination next to try next-link logic
- Progressive scroll to trigger lazy load and extract all product cards
- Save a single CSV with columns:
    site, category, section, product_id, title, price, image_url, product_url

Usage:
    pip install selenium webdriver-manager pandas
    python rainbow_scraper.py [--headless] [--out rainbow_products.csv] [--max-scrolls 30] [--pagination scroll|next]

Notes:
 - The script is made resilient to minor DOM changes but sites change over time.
 - If you want to target a specific area name other than "Johar", edit the function auto_select_location.
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

BASE = "https://rainbowcc.com.pk"


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


def auto_select_location(driver, wait, preferred_city="Lahore", preferred_area_keyword="11111Johar Town", max_retries=5):
    """
    Select location from Rainbow popup: City (Lahore) and Area (JoharTown).
    Clicks the MuiAutocomplete dropdowns and selects options.
    Retries up to max_retries times if elements aren't ready.
    """
    print(f"[Location] attempting auto-select: {preferred_city} / {preferred_area_keyword}...")
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[Location] Attempt {attempt}/{max_retries}")
            time.sleep(1.5)
            
            # Step 1: Click city dropdown (first MuiAutocomplete-popupIndicator)
            print("[Location] Opening city dropdown...")
            city_dropdown_buttons = driver.find_elements(By.CSS_SELECTOR, "button.MuiAutocomplete-popupIndicator[aria-label='Open']")
            
            if not city_dropdown_buttons:
                print(f"[Location] City dropdown not found, waiting and retrying... ({attempt}/{max_retries})")
                time.sleep(2.0)
                continue
            
            if len(city_dropdown_buttons) >= 1:
                try:
                    city_dropdown_buttons[0].click()
                    time.sleep(0.8)
                    print("[Location] Clicked city dropdown")
                except Exception as e:
                    print(f"[Location] Failed to click city dropdown: {e}")
                    time.sleep(2.0)
                    continue
            
            # Step 2: Select city from listbox
            print(f"[Location] Looking for city: {preferred_city}...")
            city_options = driver.find_elements(By.CSS_SELECTOR, "ul[role='listbox'] li, li[role='option']")
            
            if not city_options:
                print(f"[Location] City options not loaded, retrying... ({attempt}/{max_retries})")
                time.sleep(2.0)
                continue
            
            clicked_city = False
            for option in city_options:
                try:
                    text = option.text.strip()
                    if preferred_city.lower() in text.lower():
                        driver.execute_script("arguments[0].scrollIntoView(true);", option)
                        time.sleep(0.2)
                        option.click()
                        clicked_city = True
                        print(f"[Location] Selected city: {text}")
                        break
                except:
                    continue
            
            if not clicked_city and city_options:
                # Fallback: click first option
                try:
                    city_options[0].click()
                    print(f"[Location] Selected first city (fallback): {city_options[0].text}")
                except:
                    pass
            
            time.sleep(0.8)
            
            # Step 3: Click area dropdown (second MuiAutocomplete-popupIndicator)
            print("[Location] Opening area dropdown...")
            area_dropdown_buttons = driver.find_elements(By.CSS_SELECTOR, "button.MuiAutocomplete-popupIndicator[aria-label='Open']")
            
            if len(area_dropdown_buttons) < 2:
                print(f"[Location] Area dropdown not found, retrying... ({attempt}/{max_retries})")
                time.sleep(2.0)
                continue
            
            if len(area_dropdown_buttons) >= 2:
                try:
                    area_dropdown_buttons[1].click()
                    time.sleep(0.8)
                    print("[Location] Clicked area dropdown")
                except Exception as e:
                    print(f"[Location] Failed to click area dropdown: {e}")
                    time.sleep(2.0)
                    continue
            
            # Step 4: Select area from listbox
            print(f"[Location] Looking for area: {preferred_area_keyword}...")
            area_options = driver.find_elements(By.CSS_SELECTOR, "ul[role='listbox'] li, li[role='option']")
            
            if not area_options:
                print(f"[Location] Area options not loaded, retrying... ({attempt}/{max_retries})")
                time.sleep(2.0)
                continue
            
            clicked_area = False
            for option in area_options:
                try:
                    text = option.text.strip()
                    if preferred_area_keyword.lower() in text.lower():
                        driver.execute_script("arguments[0].scrollIntoView(true);", option)
                        time.sleep(0.2)
                        option.click()
                        clicked_area = True
                        print(f"[Location] Selected area: {text}")
                        break
                except:
                    continue
            
            if not clicked_area and area_options:
                # Fallback: click first option
                try:
                    area_options[0].click()
                    print(f"[Location] Selected first area (fallback): {area_options[0].text}")
                except:
                    pass
            
            time.sleep(0.8)
            
            # Step 5: Click the Select button
            print("[Location] Clicking Select button...")
            select_buttons = driver.find_elements(By.XPATH, "//button[contains(., 'Select')]")
            
            if not select_buttons:
                print(f"[Location] Select button not found, retrying... ({attempt}/{max_retries})")
                time.sleep(2.0)
                continue
            
            for btn in select_buttons:
                try:
                    disabled = btn.get_attribute("disabled")
                    if disabled is None:
                        driver.execute_script("arguments[0].scrollIntoView(true);", btn)
                        time.sleep(0.2)
                        btn.click()
                        print("[Location] Clicked Select button")
                        time.sleep(2.0)
                        break
                except Exception as e:
                    continue
            
            print(f"[Location] Location selection completed successfully on attempt {attempt}")
            time.sleep(1.0)
            return  # Success - exit the function
            
        except Exception as e:
            print(f"[Location] Error during location selection attempt {attempt}: {e}")
            if attempt < max_retries:
                print(f"[Location] Waiting 2 seconds before retry...")
                time.sleep(2.0)
            continue
    
    print(f"[Location] WARNING: Failed to select location after {max_retries} attempts")
    time.sleep(1.0)


def get_category_links(driver, wait):
    """
    From the homepage carousel/swiper get category links (anchors with href starting /catalog).
    Returns list of (title, href)
    """
    driver.get(BASE)
    time.sleep(2.5)  # Increased wait for page to fully load
    auto_select_location(driver, wait)

    time.sleep(1.5)  # Wait for content to load after location selection
    
    # Try multiple selectors to find category links
    anchors = driver.find_elements(By.CSS_SELECTOR, "div.swiper-slide a[href^='/catalog/'], a[href^='/catalog/'], div.MuiBox-root a[href^='/catalog/']")
    
    # If no links found, try scrolling down to load more content
    if not anchors:
        print("[Categories] No links found initially, scrolling to load content...")
        driver.execute_script("window.scrollTo(0, 500);")
        time.sleep(1.0)
        driver.execute_script("window.scrollTo(0, 1000);")
        time.sleep(1.0)
        anchors = driver.find_elements(By.CSS_SELECTOR, "div.swiper-slide a[href^='/catalog/'], a[href^='/catalog/'], div.MuiBox-root a[href^='/catalog/']")
    
    links = []
    for a in anchors:
        try:
            href = a.get_attribute("href") or a.get_attribute("data-href") or ""
            title = (a.get_attribute("title") or a.text or "").strip()
            if href:
                full = href if href.startswith("http") else urljoin(BASE, href)
                links.append((title or full.rstrip("/").split("/")[-1].replace("-", " ").title(), full))
        except:
            continue
    seen = set()
    uniq = []
    for t, h in links:
        if h not in seen:
            seen.add(h)
            uniq.append((t, h))
    print(f"[Categories] found {len(uniq)} category links")
    return uniq


def progressive_scroll(driver, pause=0.8, max_rounds=30):
    """
    Scroll to bottom repeatedly to lazy-load products.
    Returns list of product card elements found after scrolling.

    NOTE: This implements **scroll-based loading** (no 'Next page').
    """
    print("[Pagination] using SCROLL to load items (no 'Next Page').")
    last_count = -1
    stable = 0
    rounds = 0
    while rounds < max_rounds and stable < 3:
        rounds += 1
        try:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        except:
            pass
        time.sleep(pause)
        cards = driver.find_elements(By.CSS_SELECTOR, "div.hazle-product-item_product_item__FSm1N, div[id^='product-item-']")
        cur = len(cards)
        if cur == last_count:
            stable += 1
        else:
            stable = 0
            last_count = cur
    time.sleep(0.6)
    return driver.find_elements(By.CSS_SELECTOR, "div.hazle-product-item_product_item__FSm1N, div[id^='product-item-']")


def extract_product_info(card):
    """Extract product fields from a product card element."""
    title = ""
    product_url = ""
    image_url = ""
    price = None
    product_id = ""
    quantity = ""

    try:
        try:
            h4 = card.find_element(By.CSS_SELECTOR, "h4[title], h4")
            title = (h4.get_attribute("title") or h4.text or "").strip()
        except:
            try:
                a = card.find_element(By.CSS_SELECTOR, "a")
                title = (a.get_attribute("title") or a.text or "").strip()
                product_url = a.get_attribute("href") or ""
            except:
                title = (card.text or "").splitlines()[0].strip()
    except:
        pass

    if not product_url:
        try:
            a = card.find_element(By.CSS_SELECTOR, "a")
            href = a.get_attribute("href") or ""
            if href:
                product_url = href if href.startswith("http") else urljoin(BASE, href)
        except:
            pass

    try:
        img = card.find_element(By.CSS_SELECTOR, "img")
        src = img.get_attribute("src") or img.get_attribute("data-src") or img.get_attribute("data-original") or ""
        if src:
            image_url = src if src.startswith("http") else urljoin(BASE, src)
    except:
        image_url = ""

    try:
        try:
            price_el = card.find_element(By.CSS_SELECTOR, "div.hazle-product-item_product_item_price_label__ET_we span, div.hazle-product-item_product_item_price_label__ET_we")
            ptxt = price_el.text or ""
            price = parse_price_from_string(ptxt)
        except:
            try:
                txt = card.text or ""
                m = re.search(r'(?:Rs\.?|PKR)\s*\d[\d,]*(?:\.\d+)?', txt)
                if m:
                    price = parse_price_from_string(m.group(0))
                else:
                    inner = card.get_attribute("innerHTML") or ""
                    m2 = re.search(r'data-price-amount\s*=\s*["\']?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)["\']?', inner)
                    if m2:
                        price = parse_price_from_string(m2.group(1))
            except:
                price = None
    except:
        price = None
    
    # Extract quantity/size
    try:
        # Look for p tag with class hazle-product-item_product_item_description__ejRDa
        qty_el = card.find_element(By.CSS_SELECTOR, "p.hazle-product-item_product_item_description__ejRDa")
        quantity = (qty_el.get_attribute("title") or qty_el.text or "").strip()
    except:
        quantity = ""

    try:
        pid = card.get_attribute("id") or ""
        if pid and pid.startswith("product-item-"):
            product_id = pid.replace("product-item-", "").strip()
    except:
        product_id = ""

    if isinstance(title, str):
        title = " ".join(title.split())

    return {
        "product_id": product_id,
        "title": title,
        "price": price,
        "quantity": quantity,
        "image_url": image_url,
        "product_url": product_url
    }


def get_section_labels(driver):
    """
    Returns list of (label_text, label_element) for "Sections" radio options on category page.
    Only picks items with radio buttons (MuiRadio), ignoring checkboxes (like Brands).
    Clicks "See more" to reveal all hidden sections before extracting.
    """
    labels = []
    try:
        # Wait a bit for the sections to load
        time.sleep(1.5)
        
        # Find all list items under the "Sections" MuiList
        # Look for labels that contain radio buttons (not checkboxes)
        section_lists = driver.find_elements(By.CSS_SELECTOR, "ul.MuiList-root.MuiList-padding")
        
        print(f"[DEBUG] Found {len(section_lists)} MuiList-root elements")
        
        for ul in section_lists:
            try:
                # Check if this list has "Sections" header
                headers = ul.find_elements(By.CSS_SELECTOR, "li.MuiListSubheader-root span")
                if headers:
                    header_text = headers[0].text.strip()
                    print(f"[DEBUG] Found list with header: '{header_text}'")
                    
                    if header_text.upper() == "SECTIONS":
                        # First, click "See more" to reveal all sections (like FISH)
                        try:
                            # The "See more" link appears right after the ul.MuiList-root closes
                            # Find it using the parent and next sibling
                            parent = ul.find_element(By.XPATH, "..")
                            see_more_links = parent.find_elements(By.CSS_SELECTOR, "a.MuiLink-root")
                            for link in see_more_links:
                                link_text = link.text.strip().lower()
                                if "see more" in link_text:
                                    print("[DEBUG] Clicking 'See more' to reveal all sections")
                                    driver.execute_script("arguments[0].click();", link)
                                    time.sleep(1.0)  # Wait for sections to expand
                                    break
                        except Exception as e:
                            print(f"[DEBUG] No 'See more' found for sections (might already be expanded): {e}")
                        
                        # Now get all label elements that contain radio buttons
                        radio_labels = ul.find_elements(By.CSS_SELECTOR, "li.MuiListItem-root label")
                        print(f"[DEBUG] Found {len(radio_labels)} labels in Sections list")
                        
                        for label in radio_labels:
                            try:
                                # Verify it has a radio button (not checkbox)
                                radio_inputs = label.find_elements(By.CSS_SELECTOR, "input[type='radio']")
                                if radio_inputs:
                                    text = label.text.strip()
                                    print(f"[DEBUG] Found section with radio: '{text}'")
                                    if text and len(text) > 1 and len(text) < 60:
                                        labels.append((text, label))
                            except Exception as e:
                                # Skip if no radio input found (might be checkbox)
                                continue
            except Exception as e:
                print(f"[DEBUG] Error processing list: {e}")
                continue
    except Exception as e:
        print(f"[DEBUG] Error in get_section_labels: {e}")
        pass

    # Dedupe
    seen = set()
    uniq = []
    for t, el in labels:
        if t not in seen and t.lower() != "sections":
            seen.add(t)
            uniq.append((t, el))
    
    print(f"[DEBUG] Returning {len(uniq)} unique sections")
    return uniq


def find_next_page_link(driver):
    """
    Attempt to find a 'Next Page' link (used only if pagination mode == 'next').
    """
    try:
        elements = driver.find_elements(By.CSS_SELECTOR, "a.amscroll-next-page, a[rel='next'], a.next, a.btn.btn-primary")
        for e in elements:
            try:
                txt = (e.text or "").strip().lower()
                if "next" in txt or "›" in txt or "»" in txt:
                    href = e.get_attribute("href")
                    if href:
                        return urljoin(BASE, href)
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
                    return urljoin(BASE, href)
            except:
                continue
    except:
        pass

    return None


def scrape_category_sections(driver, wait, category_title, category_url, pagination_mode='scroll', max_scrolls=30, max_pages=4):
    """
    On a category page, iterate sections (radio) and scrape products for each section.
    pagination_mode: 'scroll' (default) or 'next' (attempt next-page links)
    """
    results = []
    driver.get(category_url)
    time.sleep(1.0)

    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "ul.MuiList-root, div.hazle-product-item_product_item__FSm1N")), timeout=6)
    except:
        time.sleep(0.7)

    sections = get_section_labels(driver)
    if not sections:
        print(f"[{category_title}] no sections found, scraping all products on page (using {pagination_mode})")
        if pagination_mode == 'scroll':
            cards = progressive_scroll(driver, pause=0.8, max_rounds=max_scrolls)
            for c in cards:
                info = extract_product_info(c)
                if not info.get("title"):
                    continue
                results.append({
                    "site": "Rainbow",
                    "category": category_title,
                    "section": "",
                    "product_id": info.get("product_id") or "",
                    "title": info.get("title"),
                    "price": info.get("price"),
                    "quantity": info.get("quantity") or "",
                    "image_url": info.get("image_url"),
                    "product_url": info.get("product_url")
                })
            return results
        else:
            # try naive next-page loop (rare for this site)
            page = 1
            current = category_url
            while page <= max_pages and current:
                driver.get(current)
                time.sleep(1.0)
                cards = driver.find_elements(By.CSS_SELECTOR, "div.hazle-product-item_product_item__FSm1N, div[id^='product-item-']")
                for c in cards:
                    info = extract_product_info(c)
                    if not info.get("title"):
                        continue
                    results.append({
                        "site": "Rainbow",
                        "category": category_title,
                        "section": "",
                        "product_id": info.get("product_id") or "",
                        "title": info.get("title"),
                        "price": info.get("price"),
                        "quantity": info.get("quantity") or "",
                        "image_url": info.get("image_url"),
                        "product_url": info.get("product_url")
                    })
                next_href = find_next_page_link(driver)
                if not next_href:
                    break
                current = next_href
                page += 1
            return results

    for idx, (sec_name, sec_el) in enumerate(sections):
        try:
            print(f"[{category_title}] selecting section '{sec_name}' (pagination: {pagination_mode})")
            
            # First section is already loaded on the category page, no need to navigate
            if idx == 0:
                print(f"[{category_title}] First section '{sec_name}' already loaded, scraping directly...")
            else:
                # For subsequent sections: click the radio button to navigate
                print(f"[{category_title}] clicking section radio button")
                try:
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", sec_el)
                    time.sleep(0.3)
                    driver.execute_script("arguments[0].click();", sec_el)
                    time.sleep(1.5)  # Wait for page to load after section change
                    print(f"[{category_title}] navigated to section URL: {driver.current_url}")
                except Exception as e:
                    print(f"[{category_title}] Error clicking section: {e}")

            
            # No need to click "See more" for products - it's already clicked in get_section_labels
            # to reveal all sections (like FISH). Products are loaded when section is clicked.

            if pagination_mode == 'scroll':
                cards = progressive_scroll(driver, pause=0.7, max_rounds=max_scrolls)
                print(f"[{category_title} - {sec_name}] found {len(cards)} product cards via SCROLL")
                seen_local = set()
                for c in cards:
                    try:
                        info = extract_product_info(c)
                        title = info.get("title") or ""
                        if not title:
                            continue
                        dedupe = info.get("product_id") or info.get("product_url") or title.lower()
                        if dedupe in seen_local:
                            continue
                        seen_local.add(dedupe)
                        results.append({
                            "site": "Rainbow",
                            "category": category_title,
                            "section": sec_name,
                            "product_id": info.get("product_id") or "",
                            "title": title,
                            "price": info.get("price"),
                            "quantity": info.get("quantity") or "",
                            "image_url": info.get("image_url"),
                            "product_url": info.get("product_url")
                        })
                    except:
                        continue
                
                # Display first 2 rows after section completes
                if results:
                    import pandas as pd
                    section_df = pd.DataFrame([r for r in results if r['section'] == sec_name])
                    if not section_df.empty:
                        print(f"\n[{category_title} - {sec_name}] First 2 products:")
                        print(section_df[['title', 'price', 'quantity']].head(2).to_string(index=False))
                        print()
            else:
                # pagination_mode == 'next'
                page = 1
                current = driver.current_url
                seen_local = set()
                while page <= 6 and current:
                    driver.get(current)
                    time.sleep(0.9)
                    cards = driver.find_elements(By.CSS_SELECTOR, "div.hazle-product-item_product_item__FSm1N, div[id^='product-item-']")
                    for c in cards:
                        info = extract_product_info(c)
                        if not info.get("title"):
                            continue
                        dedupe = info.get("product_id") or info.get("product_url") or info.get("title").lower()
                        if dedupe in seen_local:
                            continue
                        seen_local.add(dedupe)
                        results.append({
                            "site": "Rainbow",
                            "category": category_title,
                            "section": sec_name,
                            "product_id": info.get("product_id") or "",
                            "title": info.get("title"),
                            "price": info.get("price"),
                            "quantity": info.get("quantity") or "",
                            "image_url": info.get("image_url"),
                            "product_url": info.get("product_url")
                        })
                    next_href = find_next_page_link(driver)
                    if not next_href:
                        break
                    current = next_href
                    page += 1
                
                # Display first 2 rows after section completes
                if results:
                    import pandas as pd
                    section_df = pd.DataFrame([r for r in results if r['section'] == sec_name])
                    if not section_df.empty:
                        print(f"\n[{category_title} - {sec_name}] First 2 products:")
                        print(section_df[['title', 'price', 'quantity']].head(2).to_string(index=False))
                        print()

        except Exception as e:
            print(f"[{category_title}] error with section '{sec_name}': {e}")
            continue

    return results


def scrape_all_categories(headless=False, max_scrolls=30, out="rainbow_products.csv", pagination='scroll'):
    # Use single driver instance for entire scraping session
    driver = create_driver(headless=headless)
    wait = WebDriverWait(driver, 12)
    
    all_results = []
    seen_global = set()
    
    try:
        categories = get_category_links(driver, wait)
        
        if not categories:
            print("[!] No categories found on home page - exiting")
            return all_results

        # Display all categories and let user choose
        print("\n" + "="*60)
        print("Available Categories:")
        print("="*60)
        for idx, (cat_title, cat_href) in enumerate(categories, start=1):
            print(f"{idx}. {cat_title}")
        print("="*60)
        
        # Get user input
        print("\nEnter category numbers to scrape (comma-separated, or 'all' for all categories):")
        print("Example: 1,3,5  or  all")
        user_input = input("Your choice: ").strip().lower()
        
        # Determine which categories to scrape
        if user_input == 'all':
            selected_categories = categories
            print(f"\n[Selected] Scraping all {len(categories)} categories")
        else:
            try:
                selected_indices = [int(x.strip()) for x in user_input.split(',')]
                selected_categories = [(categories[i-1][0], categories[i-1][1]) for i in selected_indices if 1 <= i <= len(categories)]
                print(f"\n[Selected] Scraping {len(selected_categories)} categories: {', '.join([cat[0] for cat in selected_categories])}")
            except (ValueError, IndexError) as e:
                print(f"[Error] Invalid input: {e}. Scraping all categories instead.")
                selected_categories = categories

        for idx, (cat_title, cat_href) in enumerate(selected_categories, start=1):
            print(f"\n=== ({idx}/{len(selected_categories)}) Category: {cat_title} ===")
            try:
                cat_results = scrape_category_sections(driver, wait, cat_title, cat_href, pagination_mode=pagination, max_scrolls=max_scrolls)
            except Exception as e:
                print(f"[!] error scraping category {cat_title}: {e}")
                cat_results = []

            for r in cat_results:
                key = (r.get("site"), r.get("product_id") or r.get("product_url") or r.get("title"))
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

            time.sleep(0.5)

    finally:
        # Close driver only at the very end
        driver.quit()

    if all_results:
        df = pd.DataFrame(all_results)
        cols = ["site", "category", "section", "product_id", "title", "price", "quantity", "image_url", "product_url"]
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        df = df[cols]
        df.to_csv(out, index=False, encoding="utf-8")
        print(f"\n[+] Saved {len(df)} products to {out}")
    else:
        print("[!] No products scraped; CSV not written.")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rainbow (rainbowcc.com.pk) category/section/product scraper")
    parser.add_argument("--headless", action="store_true", help="Run Chrome headless")
    parser.add_argument("--out", default="rainbow_products.csv", help="Output CSV filename")
    parser.add_argument("--max-scrolls", type=int, default=30, help="Max scroll rounds per section (default=30)")
    parser.add_argument("--pagination", choices=["scroll", "next"], default="scroll", help="Pagination mode: 'scroll' (default) or 'next'")
    args = parser.parse_args()

    scrape_all_categories(headless=args.headless, max_scrolls=args.max_scrolls, out=args.out, pagination=args.pagination)
