#!/usr/bin/env python3
"""
Bing Copilot scraper using signed-in Chrome profile to bypass bot detection.

This uses your actual Chrome profile with logged-in session cookies,
which significantly increases success rate against bot detection.
"""
import os
import random
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.action_chains import ActionChains
import undetected_chromedriver as uc

# Configuration
QUERY_FILE = '/Users/stephaniekim/development/seoaso/geoseo_analysis/queries/seo_aso_prompts.txt'
BING_AI_OUTPUT_DIR = '../data/raw/html/bing_ai_search_results_html_profile'
CHROME_PROFILE_PATH = '/Users/stephaniekim/Library/Application Support/Google/Chrome'
PROFILE_DIRECTORY = 'Profile 3'  # Your main Chrome profile

# Create output directory
os.makedirs(BING_AI_OUTPUT_DIR, exist_ok=True)

# Read queries
with open(QUERY_FILE, 'r', encoding='utf-8') as f:
    queries = [line.strip() for line in f if line.strip()]

print(f"="*80)
print(f"BING COPILOT SCRAPER - SIGNED-IN PROFILE MODE")
print(f"="*80)
print(f"Total queries: {len(queries)}")
print(f"Output directory: {BING_AI_OUTPUT_DIR}")
print(f"Chrome profile: {PROFILE_DIRECTORY}")
print(f"Delay between queries: 90-180 seconds (very slow to avoid detection)")
print(f"="*80)
print(f"\nIMPORTANT: Make sure you're logged into Bing in {PROFILE_DIRECTORY}")
print(f"="*80)

def random_sleep(min_sec, max_sec, message=None):
    """Sleep with optional message."""
    delay = random.uniform(min_sec, max_sec)
    if message:
        print(f"  → {message} ({delay:.1f}s)")
    time.sleep(delay)

def simulate_human_mouse_movement(driver):
    """Simulate realistic mouse movements."""
    try:
        actions = ActionChains(driver)
        for _ in range(random.randint(2, 4)):
            x_offset = random.randint(50, 500)
            y_offset = random.randint(50, 300)
            actions.move_by_offset(x_offset, y_offset)
            actions.perform()
            random_sleep(0.1, 0.3)
    except Exception:
        pass

def simulate_human_scrolling(driver):
    """Simulate realistic scrolling behavior."""
    try:
        # Scroll down in small increments
        for i in range(random.randint(3, 6)):
            scroll_amount = random.randint(200, 400)
            driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
            random_sleep(0.5, 1.5)

        # Occasionally scroll back up
        if random.random() < 0.3:
            driver.execute_script("window.scrollBy(0, -300);")
            random_sleep(0.5, 1.0)

        # Scroll to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        random_sleep(1.5, 3.0)
    except Exception:
        pass

# Initialize driver with user profile
print(f"\n→ Initializing Chrome with your signed-in profile...")
options = uc.ChromeOptions()

# Use actual Chrome profile
options.add_argument(f'--user-data-dir={CHROME_PROFILE_PATH}')
options.add_argument(f'--profile-directory={PROFILE_DIRECTORY}')

# Other options
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--disable-blink-features=AutomationControlled')

print(f"  ✓ Using profile: {PROFILE_DIRECTORY}")

driver = uc.Chrome(options=options)

# Set additional properties
driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
    'source': '''
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined
        });
    '''
})

def fetch_bing_ai_page(query: str, query_num: int, total: int):
    """Fetch Bing Copilot page using signed-in profile."""
    copilot_url = f"https://www.bing.com/copilotsearch?q={query.replace(' ', '+')}"

    try:
        print(f"\n{'='*60}")
        print(f"[{query_num}/{total}] Query: {query}")
        print(f"{'='*60}")

        # Visit Bing homepage first occasionally (more realistic)
        if query_num == 1 or random.random() < 0.15:
            print("→ Visiting Bing homepage first...")
            driver.get("https://www.bing.com")
            random_sleep(3, 6, "Waiting on homepage")
            simulate_human_mouse_movement(driver)

        print(f"→ Loading Bing Copilot URL...")
        driver.get(copilot_url)

        # Wait longer for page to initialize
        random_sleep(5, 10, "Initial page load")

        # Simulate human mouse movement
        simulate_human_mouse_movement(driver)

        wait = WebDriverWait(driver, 60)  # Longer timeout

        # Wait for Copilot elements
        print("→ Waiting for Copilot interface...")
        try:
            wait.until(EC.presence_of_element_located((By.ID, "b_copilot_search")))
            print("  ✓ Copilot container loaded")

            random_sleep(3, 6, "Waiting for canvas")

            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "b_cs_canvas")))
            print("  ✓ Canvas area loaded")

            # Wait longer for AI response
            random_sleep(8, 15, "Waiting for AI response")

            # Check for retry button
            try:
                retry_button = driver.find_element(By.CSS_SELECTOR, "button.ca_action_btn, button[aria-label*='Retry']")
                if retry_button.is_displayed():
                    print("  → Found 'Retry' button - clicking...")
                    retry_button.click()
                    random_sleep(10, 20, "Waiting after retry")
            except:
                pass

            # Check for error messages
            page_source = driver.page_source
            if "no results" in page_source.lower() or "check your spelling" in page_source.lower():
                print("  ⚠️  WARNING: Bot detection - 'no results' message")
            else:
                print("  ✓ No error messages detected")

            # Simulate human behavior
            print("→ Simulating human behavior...")
            simulate_human_scrolling(driver)
            random_sleep(3, 6, "Reading content")

            # Extract iframe content
            print("→ Extracting iframe content...")
            try:
                iframe = driver.find_element(By.CSS_SELECTOR, "iframe[aria-label*='result container']")
                driver.switch_to.frame(iframe)
                random_sleep(3, 6, "Loading iframe content")
                iframe_html = driver.page_source
                driver.switch_to.default_content()
                print(f"  ✓ Iframe content extracted ({len(iframe_html)/1024:.1f} KB)")
            except Exception as e:
                print(f"  ⚠️  Could not extract iframe: {e}")
                iframe_html = ""

        except TimeoutException:
            print("  ⚠️ Timeout waiting for Copilot elements")
            iframe_html = ""

        # Get page HTML and embed iframe content
        page_html = driver.page_source

        if iframe_html:
            page_html = page_html.replace(
                '</body>',
                f'<!-- IFRAME_CONTENT_START -->{iframe_html}<!-- IFRAME_CONTENT_END --></body>'
            )

        file_size_kb = len(page_html) / 1024

        # Verify content
        has_copilot = "b_copilot_search" in page_html and "b_cs_canvas" in page_html
        has_error = "no results" in page_html.lower() or "check your spelling" in page_html.lower()

        if has_copilot and not has_error:
            print(f"  ✓ Copilot content verified ({file_size_kb:.1f} KB)")
            status = "✓ SUCCESS"
        elif has_copilot and has_error:
            print(f"  ⚠️  Copilot loaded but shows error ({file_size_kb:.1f} KB)")
            status = "⚠️  BOT DETECTED"
        else:
            print(f"  ✗ Copilot elements not found ({file_size_kb:.1f} KB)")
            status = "✗ FAILED"

        # Save HTML
        out_path = os.path.join(BING_AI_OUTPUT_DIR, f"{query.replace(' ', '_')}_bing_ai.html")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(page_html)

        print(f"  → Saved: {os.path.basename(out_path)}")
        print(f"  → Status: {status}")

        return status

    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return "✗ ERROR"

# Main scraping loop
try:
    success_count = 0
    bot_detected_count = 0
    error_count = 0

    for idx, query in enumerate(queries, 1):
        try:
            status = fetch_bing_ai_page(query, idx, len(queries))

            if "SUCCESS" in status:
                success_count += 1
            elif "BOT DETECTED" in status:
                bot_detected_count += 1
            else:
                error_count += 1

            # Much longer delays between queries (profile mode is slower but more reliable)
            if idx < len(queries):
                delay = random.uniform(90, 180)  # 1.5 to 3 minutes
                print(f"\n→ Waiting {delay:.1f} seconds before next query...")
                print(f"   (Progress: {idx}/{len(queries)} | Success: {success_count} | Bot: {bot_detected_count} | Error: {error_count})")
                time.sleep(delay)

        except KeyboardInterrupt:
            print(f"\n\n⚠️  Scraping interrupted by user at query {idx}/{len(queries)}")
            break
        except Exception as e:
            print(f"❌ Failed to process query {idx}: {e}")
            error_count += 1
            continue

    # Final summary
    print(f"\n{'='*80}")
    print(f"SCRAPING COMPLETE")
    print(f"{'='*80}")
    print(f"Total queries: {len(queries)}")
    print(f"Successful: {success_count} (✓)")
    print(f"Bot detected: {bot_detected_count} (⚠️)")
    print(f"Errors: {error_count} (✗)")
    print(f"Output directory: {BING_AI_OUTPUT_DIR}")

    if success_count > 50:
        print(f"\n✓ SUCCESS: Signed-in profile significantly improved success rate!")
    elif bot_detected_count > success_count:
        print(f"\n⚠️  WARNING: Bot detection still occurring frequently")
        print(f"    Consider increasing delays further or manual collection")

except KeyboardInterrupt:
    print(f"\n\n⚠️  Scraping interrupted by user")

finally:
    print(f"\n→ Closing browser...")
    driver.quit()
    print(f"✓ Browser closed")
