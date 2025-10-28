# carrefour_search_scraper.py
# Usage:
#   python carrefour_search_scraper.py pepsi 2
#   (search term "pepsi", scrape up to 2 pages)
#
# Requirements:
#   pip install selenium webdriver-manager pandas
#
# Notes:
# - The script opens a browser so you can interact if the site asks for location.
# - It scrapes product cards on the search result page and tries to go to the "Next" page up to max_pages.
# - If selectors change, update the CSS selectors in the extraction block.

import sys, time, csv, re
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

BASE = "https://www.carrefour.pk"
SEARCH_BASE = BASE + "/mafpak/en/v4/search?keyword="

def create_driver(headless=False):
    opts = webdriver.ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)

def parse_price_from_parts(int_text, frac_text):
    """
    Given integer part (e.g. '160') and fractional part (e.g. '.00' or '00'), return float 160.00
    """
    try:
        int_part = re.sub(r'[^\d]', '', int_text or "")
        frac = ""
        if frac_text:
            frac = re.search(r'(\d+)', frac_text)
            frac = frac.group(1) if frac else ""
        if frac:
            s = int_part + "." + frac
        else:
            s = int_part
        return float(s) if s else None
    except:
        return None

def parse_price_fallback(text):
    if not text:
        return None
    s = re.sub(r'[^\d.]', '', text.replace(',', ''))
    try:
        return float(s) if s else None
    except:
        return None

def scrape_search(term="pepsi", max_pages=2, headless=False, out_csv=None):
    driver = create_driver(headless=headless)
    wait = WebDriverWait(driver, 15)
    out_csv = out_csv or f"carrefour_{term}.csv"
    results = []
    try:
        url = SEARCH_BASE + term
        driver.get(url)
        print("[i] Opened:", url)
        print("[i] If the site asks for location or any popup, handle it in the opened browser now.")
        time.sleep(1)
        input("Press Enter after search results are visible in the browser (or if none required) ...")

        page = 1
        while page <= max_pages:
            # wait for product cards to appear
            # product cards in your HTML are <ul class="css-1omnv59">; use that selector
            product_selector = "ul.css-1omnv59"
            wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, product_selector)))
            time.sleep(0.6)  # let JS finish

            cards = driver.find_elements(By.CSS_SELECTOR, product_selector)
            print(f"[i] Page {page}: found {len(cards)} product cards")

            for card in cards:
                try:
                    # product name and product_url
                    try:
                        name_el = card.find_element(By.CSS_SELECTOR, "a[data-testid='product_name']")
                        name = name_el.text.strip()
                        product_href = name_el.get_attribute("href") or name_el.get_attribute("data-href")
                        product_url = urljoin(BASE, product_href) if product_href else ""
                    except:
                        # fallback to any anchor with title
                        try:
                            a = card.find_element(By.CSS_SELECTOR, "a[title]")
                            name = a.get_attribute("title") or a.text.strip()
                            href = a.get_attribute("href")
                            product_url = urljoin(BASE, href) if href else ""
                        except:
                            name = ""
                            product_url = ""

                    # image
                    try:
                        img = card.find_element(By.CSS_SELECTOR, "img[data-testid='product_image_main']")
                        img_src = img.get_attribute("src")
                        image_url = img_src if img_src.startswith("http") else urljoin(BASE, img_src)
                    except:
                        image_url = ""

                    # price: preferred parts
                    price = None
                    try:
                        int_part_el = card.find_element(By.CSS_SELECTOR, "div.css-14zpref")
                        frac_part_el = card.find_element(By.CSS_SELECTOR, "div.css-1pjcwg4")
                        int_text = int_part_el.text.strip()
                        frac_text = frac_part_el.text.strip()
                        price = parse_price_from_parts(int_text, frac_text)
                    except Exception:
                        # fallback: full price container text
                        try:
                            price_container = card.find_element(By.CSS_SELECTOR, "div[data-testid='product_price'], div.css-17fvam3")
                            price_txt = price_container.text.strip()
                            price = parse_price_fallback(price_txt)
                        except Exception:
                            price = None

                    results.append({
                        "name": name,
                        "price": price,
                        "image_url": image_url,
                        "product_url": product_url,
                        "page": page
                    })
                except Exception as e:
                    print("[!] error parsing card:", e)
                    continue

            # attempt to navigate to next page:
            try:
                # try finding a "Next" link or button - websites vary.
                # try anchor with rel=next
                next_el = None
                try:
                    next_el = driver.find_element(By.CSS_SELECTOR, "a[rel='next']")
                except:
                    pass
                if not next_el:
                    # try simple "Next" button text
                    try:
                        next_el = driver.find_element(By.XPATH, "//button[contains(translate(., 'NEXT', 'next'), 'next') or contains(., 'Next')]")
                    except:
                        next_el = None
                if not next_el:
                    # try pagination link with aria-label or class containing 'next'
                    try:
                        next_el = driver.find_element(By.CSS_SELECTOR, "a[class*='next'], button[class*='next'], a[aria-label='Next']")
                    except:
                        next_el = None

                if not next_el:
                    print("[i] No Next element found — stopping pagination.")
                    break

                # check if disabled
                disabled_attr = next_el.get_attribute("disabled")
                cls = (next_el.get_attribute("class") or "").lower()
                if disabled_attr or "disabled" in cls:
                    print("[i] Next is disabled — last page reached.")
                    break

                # click and wait for new cards
                print("[i] Clicking Next ...")
                before_first = None
                try:
                    before_first = driver.find_element(By.CSS_SELECTOR, product_selector).get_attribute("outerHTML")
                except:
                    before_first = None
                driver.execute_script("arguments[0].scrollIntoView(true);", next_el)
                time.sleep(0.2)
                next_el.click()
                # wait for change
                def content_changed(drv):
                    try:
                        after_first = drv.find_element(By.CSS_SELECTOR, product_selector).get_attribute("outerHTML")
                        return before_first != after_first
                    except:
                        return True
                WebDriverWait(driver, 15).until(content_changed)
                time.sleep(0.6)
                page += 1
            except Exception as e:
                print("[i] Could not navigate to next page or no more pages:", e)
                break

        # write CSV
        if results:
            keys = ["name", "price", "image_url", "product_url", "page"]
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
    max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    scrape_search(term=term, max_pages=max_pages, headless=False)
