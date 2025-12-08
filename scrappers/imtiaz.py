# imtiaz_search_scraper.py
# Usage:
#   python imtiaz_search_scraper.py pepsi 3
# where "pepsi" is the search term and 3 is max pages to scrape.
#
# Requirements:
#   pip install selenium webdriver-manager pandas

import sys
import time
import csv
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

BASE = "https://shop.imtiaz.com.pk"

def create_driver(headless=False):
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    # optional: make browser look less like automation
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--window-size=1366,900")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

def parse_price(text):
    if not text:
        return None
    import re
    # common patterns: "Rs. 145.00", "PKR 145", "145.00"
    # remove non-digit except dot and comma, then convert
    s = text.replace(',', '')  # remove thousands comma
    m = re.search(r'(\d+(?:\.\d+)?)', s)
    if m:
        try:
            return float(m.group(1))
        except:
            return None
    return None

def auto_select_location(driver, wait, prefer_city="Karachi", max_wait=8):
    """
    Attempt to automatically select a delivery city/area.
    Strategy:
      - If an 'Open' aria-label button exists for autocomplete dropdowns, click the second one (area).
      - Wait for the listbox options to appear and click the first option.
      - Find an enabled 'Select' button and click it.
      - If any step fails, function returns gracefully (script continues).
    """
    try:
        print("[i] Looking for location selection modal...")
        # small wait: the modal/popover may take a moment to appear
        time.sleep(1.0)

        # Find all 'Open' buttons for autocomplete and click the area dropdown (prefer last)
        open_buttons = driver.find_elements(By.CSS_SELECTOR, "button[aria-label='Open']")
        if open_buttons:
            # attempt to click last 'Open' button (likely Area / Sub Region)
            try:
                btn = open_buttons[-1]
                driver.execute_script("arguments[0].scrollIntoView(true);", btn)
                time.sleep(0.2)
                btn.click()
                print("[i] Clicked area dropdown button.")
            except Exception as e:
                print("[!] Could not click area dropdown button:", e)

            # wait for options to appear
            try:
                # possible selectors for list items in MUI popper/listbox
                list_item_locator = (By.XPATH, "//ul[@role='listbox']//li | //div[contains(@class,'MuiAutocomplete-popper')]//li | //li[contains(@role,'option')]")
                WebDriverWait(driver, max_wait).until(EC.presence_of_element_located(list_item_locator))
                time.sleep(0.3)
                options = driver.find_elements(*list_item_locator)
                if options:
                    # click the first visible option
                    clicked = False
                    for opt in options:
                        try:
                            if opt.is_displayed():
                                driver.execute_script("arguments[0].scrollIntoView(true);", opt)
                                time.sleep(0.15)
                                opt.click()
                                clicked = True
                                print("[i] Selected first area option from dropdown.")
                                break
                        except Exception:
                            continue
                    if not clicked:
                        print("[i] Found options but couldn't click any; continuing.")
                else:
                    print("[i] No options found in dropdown; continuing.")
            except Exception as e:
                print("[i] Timeout or error waiting for area options:", e)

        else:
            print("[i] No 'Open' dropdown buttons found — location modal may not be present.")

        # After picking area, click the 'Select' button (enabled one)
        try:
            # wait a moment for 'Select' to become enabled
            time.sleep(0.5)
            # find all buttons with text 'Select' and click the first enabled one
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
                        print("[i] Clicked 'Select' button to confirm location.")
                        break
                except Exception:
                    continue
            if not clicked_select:
                print("[i] No enabled 'Select' button found (maybe already set).")
        except Exception as e:
            print("[!] Error while trying to click Select button:", e)

        # Wait a bit for the site to process location and load products
        wait_seconds = 7
        print(f"[i] Waiting {wait_seconds} seconds for products to load after location selection...")
        time.sleep(wait_seconds)
    except Exception as e:
        # never fail the whole scraper if location handling fails
        print("[!] auto_select_location error:", e)
    finally:
        # short extra pause to let any final XHRs settle
        time.sleep(0.5)

def scrape_search(query="pepsi", max_pages=3, headless=False, out_csv="imtiaz_search_results.csv"):
    driver = create_driver(headless=headless)
    wait = WebDriverWait(driver, 15)
    results = []
    seen_ids = set()
    try:
        url = f"{BASE}/search?q={query}"
        driver.get(url)
        print(f"[i] Opened {url}")
        # automatically try to select location (no user input)
        auto_select_location(driver, wait)

        page_no = 1
        while page_no <= max_pages:
            # Wait for product cards to appear (give a bit longer because items can load dynamically)
            product_selector = "div[id^='product-item-']"
            try:
                wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, product_selector)))
            except Exception:
                # if timeout, still attempt to continue; we'll get zero cards then stop
                print("[i] Timeout waiting for product cards - continuing anyway")
            time.sleep(0.8)  # let JS finish rendering

            cards = driver.find_elements(By.CSS_SELECTOR, product_selector)
            print(f"[i] Found {len(cards)} cards on page {page_no}")

            for card in cards:
                try:
                    # product id from id attribute
                    pid_attr = card.get_attribute("id") or ""
                    pid = None
                    if pid_attr.startswith("product-item-"):
                        pid = pid_attr.replace("product-item-", "").strip()

                    # title/name: h4 with class (fallback to any h4)
                    try:
                        name_el = card.find_element(By.CSS_SELECTOR, "h4.hazle-product-item_product_item_title__wK9IT")
                    except:
                        try:
                            name_el = card.find_element(By.CSS_SELECTOR, "h4[title]")
                        except:
                            name_el = None
                    name = name_el.get_attribute("title") if name_el else (name_el.text if name_el else "")

                    # description (if any)
                    try:
                        desc_el = card.find_element(By.CSS_SELECTOR, "p.hazle-product-item_product_item_description__ejRDa")
                        description = desc_el.get_attribute("title") or desc_el.text
                    except:
                        description = ""

                    # price: prefer specific price-label div -> span
                    price = None
                    try:
                        # exact selector for the price label you pasted
                        price_span = card.find_element(By.CSS_SELECTOR, "div.hazle-product-item_product_item_price_label__ET_we span")
                        price_text = price_span.text.strip()
                        price = parse_price(price_text)
                    except Exception:
                        # fallback: any span or element that contains "Rs" (covers small variations)
                        try:
                            maybe = card.find_element(By.XPATH, ".//span[contains(., 'Rs') or contains(., 'Rs.') or contains(., 'PKR')]")
                            price = parse_price(maybe.text)
                        except Exception:
                            # final fallback: search for digits anywhere inside the card
                            try:
                                maybe2 = card.find_element(By.XPATH, ".//*[normalize-space()[string-length(.)>0]]")
                                price = parse_price(maybe2.text)
                            except Exception:
                                price = None

                    # image url
                    img_url = ""
                    try:
                        img_el = card.find_element(By.CSS_SELECTOR, "img")
                        src = img_el.get_attribute("src") or img_el.get_attribute("data-src")
                        if src:
                            img_url = urljoin(BASE, src) if src.startswith("/") else src
                    except:
                        img_url = ""

                    # product page link: may or may not exist as anchor; try to find nearest <a>
                    product_url = ""
                    try:
                        a = card.find_element(By.CSS_SELECTOR, "a")
                        href = a.get_attribute("href")
                        if href:
                            product_url = href
                    except:
                        product_url = ""

                    # avoid duplicates
                    if pid and pid in seen_ids:
                        continue
                    seen_ids.add(pid)

                    results.append({
                        "product_id": pid,
                        "name": name,
                        "description": description,
                        "price": price,
                        "image_url": img_url,
                        "product_url": product_url,
                        "page": page_no
                    })
                except Exception as e:
                    print("[!] error parsing a card:", e)
                    continue

            # attempt to find Next button and click it
            try:
                # Next button text = "Next" in the HTML you posted
                next_btn = driver.find_element(By.XPATH, "//button[normalize-space()='Next']")
                classes = (next_btn.get_attribute("class") or "")
                disabled_attr = next_btn.get_attribute("disabled")
                disabled = disabled_attr is not None or "Mui-disabled" in classes or "disabled" in classes
                if disabled:
                    print("[i] Next button disabled -> reached last page")
                    break
                # click Next, wait for products to change
                print("[i] Clicking Next ...")
                # get first card id before click to detect change
                first_before = None
                try:
                    first_before = driver.find_element(By.CSS_SELECTOR, product_selector).get_attribute("id")
                except:
                    first_before = None
                next_btn.click()
                # wait for new content: either staleness or new first item id
                def first_card_changed(driver):
                    try:
                        first_after = driver.find_element(By.CSS_SELECTOR, product_selector).get_attribute("id")
                        return first_after != first_before
                    except:
                        return True
                WebDriverWait(driver, 15).until(first_card_changed)
                time.sleep(0.6)
                page_no += 1
            except Exception as e:
                print("[i] Could not find/click Next or no more pages:", e)
                break

        # Save CSV
        if results:
            keys = ["product_id", "name", "description", "price", "image_url", "product_url", "page"]
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for r in results:
                    writer.writerow(r)
            print(f"[+] Saved {len(results)} rows to {out_csv}")
        else:
            print("[!] No results collected")

    finally:
        driver.quit()


if __name__ == "__main__":
    term = sys.argv[1] if len(sys.argv) > 1 else "pepsi"
    max_pages_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    out_csv = f"imtiaz_search_{term}.csv"
    scrape_search(query=term, max_pages=max_pages_arg, headless=False, out_csv=out_csv)
