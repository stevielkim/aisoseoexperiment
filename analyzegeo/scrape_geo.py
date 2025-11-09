import os
import random
import time
import requests
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
PERPLEXITY_OUTPUT_DIR = 'perplexity_search_results_html'
GOOGLE_AI_OUTPUT_DIR = 'google_ai_search_results_html'
BING_AI_OUTPUT_DIR = 'bing_ai_search_results_html'

# Create output directories if they don't exist
os.makedirs(PERPLEXITY_OUTPUT_DIR, exist_ok=True)
os.makedirs(GOOGLE_AI_OUTPUT_DIR, exist_ok=True)
os.makedirs(BING_AI_OUTPUT_DIR, exist_ok=True)

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

def get_more():
    # Try to click "Show more" in AI Overview if available
    try:
        show_more_btn = driver.find_element(By.XPATH, '//span[contains(text(), "Show more") or contains(text(), "More")]')
        driver.execute_script("arguments[0].click();", show_more_btn)
        time.sleep(random.uniform(2, 4))  # Wait for expanded content to load
    except Exception as e:
        print(f"No 'Show more' button found or could not click: {e}")

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
        get_more()
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
    """Load Bing SERP, switch to Copilot tab, expand the answer, save HTML."""
    search_url = f"https://www.bing.com/search?q={query.replace(' ', '+')}"
    try:
        driver.get(search_url)
        wait = WebDriverWait(driver, 20)

        # 1) Scroll to trigger Copilot tab render
        body = driver.find_element(By.TAG_NAME, "body")
        for _ in range(3):
            body.send_keys(Keys.PAGE_DOWN)
            time.sleep(random.uniform(0.8, 1.5))

        # 2) Click Copilot tab if visible
        try:
            copilot_tab = wait.until(
                EC.element_to_be_clickable((By.XPATH, '//div[contains(text(),"Copilot")]'))
            )
            copilot_tab.click()
            print("✓ Copilot tab clicked")
        except Exception:
            print("⚠️ Copilot tab not found - proceeding (may be auto-open)")

        # 3) Wait for Copilot answer block (new <cib-serp> tag or legacy .b_factrow)
        try:
            wait.until(
                EC.any_of(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "cib-serp")),
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.b_factrow"))
                )
            )
            print("✓ Copilot answer detected")
        except Exception:
            print("⚠️ Copilot answer not detected within timeout")

        # 4) Click "Show more" inside Copilot if present
        try:
            show_more = wait.until(
                EC.element_to_be_clickable(
                    (By.XPATH, '//button[contains(text(),"Show more")]')
                )
            )
            show_more.click()
            print("✓ Copilot ‘Show more’ clicked")
            time.sleep(2)  # give it a moment to render extra citations
        except Exception:
            print("ℹ️ No 'Show more' button visible")

        # 5) Final scroll to bottom so all lazy-loaded anchors render
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)

        # 6) Save page source
        out_path = os.path.join(
            BING_AI_OUTPUT_DIR, f"{query.replace(' ', '_')}_bing_ai.html"
        )
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(driver.page_source)
        print(f"✓ Saved Copilot HTML → {out_path}")

    except Exception as e:
        print(f"❌ Error fetching Bing Copilot for “{query}”: {e}")



# Collect Results for All AI Search Engines
for query in queries:
    print(f"Searching Perplexity.ai for: {query}")
    fetch_perplexity_page(query)
    print(f"Searching Google AI for: {query}")
    fetch_google_ai_page(query)
    print(f"Searching Bing AI for: {query}")
    fetch_bing_ai_page(query)
    time.sleep(random.uniform(2, 5))  # Random delay to avoid blocking

driver.quit()

print(f"All results saved to {PERPLEXITY_OUTPUT_DIR}, {GOOGLE_AI_OUTPUT_DIR}, and {BING_AI_OUTPUT_DIR}.")
