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

def scrape_search(query="pepsi", max_pages=3, headless=False, out_csv="imtiaz_search_results.csv"):
    driver = create_driver(headless=headless)
    wait = WebDriverWait(driver, 15)
    results = []
    seen_ids = set()
    try:
        url = f"{BASE}/search?q={query}"
        driver.get(url)
        print(f"[i] Opened {url}")
        print("[i] If site asks for location/branch selection, please do that in the opened browser window now.")
        input("Press Enter here after the search results are visible in the browser (or if none required) ...")

        page_no = 1
        while page_no <= max_pages:
            # Wait for product cards to appear
            # Product cards have id="product-item-<num>"
            product_selector = "div[id^='product-item-']"
            wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, product_selector)))
            time.sleep(0.5)  # let JS finish rendering

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
