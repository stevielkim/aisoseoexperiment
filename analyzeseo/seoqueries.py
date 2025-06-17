import os
import random
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# Input File
QUERY_FILE = 'data/seo_aso_prompts.txt'
GOOGLE_OUTPUT_DIR = 'google_search_results_html'
BING_OUTPUT_DIR = 'bing_search_results_html'

# Create output directories if they don't exist
os.makedirs(GOOGLE_OUTPUT_DIR, exist_ok=True)
os.makedirs(BING_OUTPUT_DIR, exist_ok=True)

# Selenium Setup
options = Options()
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_argument('--disable-gpu')
options.add_argument('--start-maximized')
options.add_argument('--disable-infobars')
options.add_argument('--disable-popup-blocking')
options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3')
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Read Queries from File
with open(QUERY_FILE, 'r') as f:
    queries = [line.strip() for line in f.readlines() if line.strip()]

# Function to Fetch and Save Full Page Source for Google

def fetch_google_page(query):
    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    try:
        driver.get(search_url)
        time.sleep(random.uniform(3, 6))  # Delay to avoid instant bot detection
        # Save the full page source for offline parsing
        output_file = os.path.join(GOOGLE_OUTPUT_DIR, f"{query.replace(' ', '_')}_google.html")
        with open(output_file, 'w', encoding='utf-8') as page_file:
            page_file.write(driver.page_source)
        print(f"Saved Google page source for query: {query}")
    except Exception as e:
        print(f"Error fetching Google results for {query}: {e}")

# Function to Fetch and Save Full Page Source for Bing

def fetch_bing_page(query):
    search_url = f"https://www.bing.com/search?q={query.replace(' ', '+')}"
    try:
        driver.get(search_url)
        time.sleep(random.uniform(3, 6))  # Delay to avoid instant bot detection
        # Save the full page source for offline parsing
        output_file = os.path.join(BING_OUTPUT_DIR, f"{query.replace(' ', '_')}_bing.html")
        with open(output_file, 'w', encoding='utf-8') as page_file:
            page_file.write(driver.page_source)
        print(f"Saved Bing page source for query: {query}")
    except Exception as e:
        print(f"Error fetching Bing results for {query}: {e}")

# Collect Results for Both Search Engines
for query in queries:
    print(f"Searching Google for: {query}")
    fetch_google_page(query)
    print(f"Searching Bing for: {query}")
    fetch_bing_page(query)
    time.sleep(random.uniform(2, 5))  # Random delay to avoid blocking

driver.quit()

print(f"All results saved to {GOOGLE_OUTPUT_DIR} and {BING_OUTPUT_DIR}.")
