import os
import random
import time
import requests
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import undetected_chromedriver as uc
from selenium.webdriver.support import expected_conditions as EC

# Input File
QUERY_FILE = '/Users/stephaniekim/development/seoaso/geoseo_analysis/queries/seo_aso_prompts.txt'

# Create timestamped directories for timeseries data
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
BASE_DIR = '../data/raw/html'
PERPLEXITY_OUTPUT_DIR = os.path.join(BASE_DIR, f'perplexity_search_results_html_{TIMESTAMP}')
GOOGLE_AI_OUTPUT_DIR = os.path.join(BASE_DIR, f'google_ai_search_results_html_{TIMESTAMP}')
BING_AI_OUTPUT_DIR = os.path.join(BASE_DIR, f'bing_ai_search_results_html_{TIMESTAMP}')

# Create output directories if they don't exist
os.makedirs(PERPLEXITY_OUTPUT_DIR, exist_ok=True)
os.makedirs(GOOGLE_AI_OUTPUT_DIR, exist_ok=True)
# Skip Bing for now - bot detection issues
# os.makedirs(BING_AI_OUTPUT_DIR, exist_ok=True)

print(f"="*80)
print(f"SCRAPING RUN - {TIMESTAMP}")
print(f"="*80)
print(f"Perplexity output: {PERPLEXITY_OUTPUT_DIR}")
print(f"Google AI output: {GOOGLE_AI_OUTPUT_DIR}")
print(f"Bing AI: SKIPPED (bot detection issues)")
print(f"="*80)

# Selenium Setup
# options = Options()
# options.add_argument('--no-sandbox')
# options.add_argument('--disable-dev-shm-usage')
# options.add_argument('--disable-blink-features=AutomationControlled')
# options.add_argument('--disable-gpu')
# options.add_argument('--start-maximized')
# options.add_argument('--disable-infobars')
# options.add_argument('--disable-popup-blocking')
# options.add_experimental_option("excludeSwitches", ["enable-automation"])
# options.add_experimental_option('useAutomationExtension', False)
# options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.199 Safari/537.36')
options = uc.ChromeOptions()
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_argument('--start-maximized')
options.add_argument('--disable-infobars')
options.add_argument('--disable-popup-blocking')
options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.199 Safari/537.36')


driver = uc.Chrome(options=options)

# driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Read Queries from File
with open(QUERY_FILE, 'r') as f:
    queries = [line.strip() for line in f.readlines() if line.strip()]

# Function to Fetch and Save Full Page Source for Perplexity.ai
def fetch_perplexity_page(query):
    try:
        driver.get("https://www.perplexity.ai")
        time.sleep(2)

    except Exception as e:
        print(f"Error adding cookies to Perplexity {e}")
    search_url = f"https://www.perplexity.ai/?q={query.replace(' ', '+')}"

    try:
        driver.get(search_url)
        time.sleep(random.uniform(5, 10))  # Delay to avoid instant bot detection
        # Save the full page source for offline parsing
        output_file = os.path.join(PERPLEXITY_OUTPUT_DIR, f"{query.replace(' ', '_')}_perplexity.html")
        with open(output_file, 'w', encoding='utf-8') as page_file:
            page_file.write(driver.page_source)
        print(f"Saved Perplexity.ai page source for query: {query}")
    except Exception as e:
        print(f"Error fetching Perplexity.ai results for {query}: {e}")


def simulate_scroll():
    # Simulate scrolling to help AI overview render
    body = driver.find_element(By.TAG_NAME, 'body')
    for _ in range(3):
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(random.uniform(1, 2))

def expand_ai_overview():
    """Click 'Show More' in AI Overview content and 'Show All' in citations sidebar."""
    clicked_something = False

    # 1. Try to expand AI Overview main content with "Show More"
    try:
        # Multiple selectors for "Show More" button in AI Overview
        show_more_selectors = [
            '//span[contains(text(), "Show More")]',
            '//button[contains(text(), "Show More")]',
            '//div[contains(@class, "show-more")]//span',
            '//span[contains(text(), "Show more")]',
            '//button[contains(., "Show more")]',
        ]

        for selector in show_more_selectors:
            try:
                show_more_btn = driver.find_element(By.XPATH, selector)
                if show_more_btn.is_displayed():
                    driver.execute_script("arguments[0].click();", show_more_btn)
                    print("  ✓ Clicked 'Show More' in AI Overview")
                    time.sleep(random.uniform(2, 4))  # Wait for content expansion
                    clicked_something = True
                    break
            except:
                continue
    except Exception:
        pass

    # 2. Try to expand citations list with "Show All"
    try:
        # Multiple selectors for "Show All" button in citations
        show_all_selectors = [
            '//span[contains(text(), "Show All")]',
            '//button[contains(text(), "Show All")]',
            '//a[contains(text(), "Show All")]',
            '//span[contains(text(), "Show all")]',
            '//button[contains(., "Show all")]',
            '//div[contains(@class, "sources")]//button[contains(., "Show")]',
        ]

        for selector in show_all_selectors:
            try:
                show_all_btn = driver.find_element(By.XPATH, selector)
                if show_all_btn.is_displayed():
                    driver.execute_script("arguments[0].click();", show_all_btn)
                    print("  ✓ Clicked 'Show All' in citations")
                    time.sleep(random.uniform(2, 4))  # Wait for citations expansion
                    clicked_something = True
                    break
            except:
                continue
    except Exception:
        pass

    if not clicked_something:
        print("  → No expand buttons found (may be fully expanded already)")

def accept_cookies():
        # Accept cookies if prompted
        wait = WebDriverWait(driver, 10)
        try:
            consent_btn = wait.until(EC.element_to_be_clickable((By.XPATH, '//button[contains(text(), "I agree")]')))
            consent_btn.click()
        except:
            pass
        time.sleep(random.uniform(2, 4))

# Function to Fetch and Save Full Page Source for Google AI Search using scrolling and JS rendering
def fetch_google_ai_page(query):
    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    try:
        driver.get(search_url)
        time.sleep(random.uniform(5, 8))  # Let initial page load
        
        accept_cookies()
        expand_ai_overview()  # Click "Show More" and "Show All"
        simulate_scroll() 
        # # Scroll down to trigger AI overview loading
        # driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        # time.sleep(random.uniform(3, 6))  # Wait for content to load

        # Save the full page source for offline parsing
        output_file = os.path.join(GOOGLE_AI_OUTPUT_DIR, f"{query.replace(' ', '_')}_google_ai.html")
        with open(output_file, 'w', encoding='utf-8') as page_file:
            page_file.write(driver.page_source)
        print(f"Saved Google AI page source for query: {query}")
    except Exception as e:
        print(f"Error fetching Google AI results for {query}: {e}")



def fetch_bing_ai_page(query: str):
    """
    Load Bing search, navigate to Copilot, wait for AI response, save HTML.

    Uses direct Copilot URL instead of trying to click tabs on SERP.
    """
    # Use correct Copilot URL (copilotsearch endpoint)
    copilot_url = f"https://www.bing.com/copilotsearch?q={query.replace(' ', '+')}"

    try:
        print(f"→ Loading Bing Copilot for: {query}")
        driver.get(copilot_url)
        wait = WebDriverWait(driver, 30)

        # Give page time to initialize
        time.sleep(random.uniform(3, 5))

        # Wait for Copilot interface elements to load (updated selectors)
        print("→ Waiting for Copilot interface...")

        try:
            # Wait for main Copilot search container
            wait.until(EC.presence_of_element_located((By.ID, "b_copilot_search")))
            print("✓ Copilot container (#b_copilot_search) loaded")

            # Wait for answer canvas to appear
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "b_cs_canvas")))
            print("✓ Canvas area (.b_cs_canvas) loaded")

            # Wait for AI disclaimer (confirms content generated)
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "b_cs_disclaimer")))
            print("✓ AI disclaimer detected")

        except Exception as e:
            print("⚠️ WARNING: No Copilot elements detected")
            print(f"   Error: {e}")

        # Wait for actual response content (AI text + citations)
        time.sleep(random.uniform(5, 8))

        # Scroll to load lazy content
        for _ in range(3):
            driver.execute_script("window.scrollBy(0, 300);")
            time.sleep(random.uniform(0.5, 1.0))

        # Try to expand "Show more" if present
        try:
            show_more_selectors = [
                '//button[contains(text(),"Show more")]',
                '//button[contains(text(),"Read more")]',
                '//a[contains(text(),"Show more")]'
            ]
            for xpath in show_more_selectors:
                try:
                    button = driver.find_element(By.XPATH, xpath)
                    driver.execute_script("arguments[0].click();", button)
                    print("✓ Clicked 'Show more'")
                    time.sleep(2)
                    break
                except:
                    continue
        except Exception:
            pass  # No "Show more" button found

        # Final scroll to ensure all content rendered
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        # Verify Copilot content before saving (updated check)
        page_html = driver.page_source
        if "b_copilot_search" in page_html and "b_cs_canvas" in page_html:
            print("✓ Copilot content verified in HTML")
        else:
            print("⚠️ WARNING: Copilot elements not found in HTML")
            print(f"   Page title: {driver.title}")

        # Save page source
        out_path = os.path.join(
            BING_AI_OUTPUT_DIR, f"{query.replace(' ', '_')}_bing_ai.html"
        )
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(page_html)

        file_size_kb = len(page_html) / 1024
        print(f"✓ Saved Copilot HTML → {out_path} ({file_size_kb:.1f} KB)")

    except Exception as e:
        print(f"❌ Error fetching Bing Copilot for '{query}': {e}")
        import traceback
        traceback.print_exc()



# Collect Results for All AI Search Engines
print(f"\nProcessing {len(queries)} queries...")
print(f"NOTE: Skipping Bing AI due to bot detection issues\n")

for idx, query in enumerate(queries, 1):
    print(f"\n[{idx}/{len(queries)}] Query: {query}")
    print("="*60)

    print(f"→ Searching Perplexity.ai...")
    fetch_perplexity_page(query)

    print(f"→ Searching Google AI...")
    fetch_google_ai_page(query)

    # Skip Bing for now
    # print(f"→ Searching Bing AI...")
    # fetch_bing_ai_page(query)

    if idx < len(queries):
        delay = random.uniform(3, 6)
        print(f"→ Waiting {delay:.1f}s before next query...")
        time.sleep(delay)

driver.quit()

print(f"\n{'='*80}")
print(f"SCRAPING COMPLETE")
print(f"{'='*80}")
print(f"Total queries processed: {len(queries)}")
print(f"Perplexity results: {PERPLEXITY_OUTPUT_DIR}")
print(f"Google AI results: {GOOGLE_AI_OUTPUT_DIR}")
print(f"\nNext steps:")
print(f"1. Run parse_geo.py with the new timestamped directories")
print(f"2. Compare results with previous runs for timeseries analysis")
