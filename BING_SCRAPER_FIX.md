# Bing Copilot Scraper Fix - December 2024

## Problem Summary

The original Bing AI scraper had a 2.8% inclusion rate (434 rows, ~12 citations), which was unrealistically low. Investigation revealed three critical issues:

### Issue 1: Outdated HTML Selectors
**Problem**: Bing changed their Copilot HTML structure between August 2024 and December 2024.
- **Old selectors** (no longer work): `<cib-serp>`, `<cib-conversation>`, `<cib-message-group>`
- **New selectors** (current): `#b_copilot_search`, `.b_cs_canvas`, `.b_cs_disclaimer`

### Issue 2: Citations in Iframe (Most Critical)
**Problem**: Bing Copilot loads AI response content in an **iframe** that is not captured by `driver.page_source`.
- The outer page contains the Copilot shell
- The actual search results with citations are loaded dynamically in `<iframe aria-label="result container">`
- Original scraper only saved the outer page HTML (missing all citations)

### Issue 3: Bot Detection
**Problem**: Bing's bot detection was blocking automated queries with "no results" errors.

## Solution Implemented

### 1. Enhanced Bot Evasion
- Random user agents, viewport sizes
- Mouse movement and scrolling simulation
- Realistic delays (45-90s between queries)
- Anti-detection JavaScript
- Automatic retry button clicking

### 2. Iframe Content Extraction (Critical Fix)
```python
# Switch into iframe to capture citations
iframe = driver.find_element(By.CSS_SELECTOR, "iframe[aria-label*='result container']")
driver.switch_to.frame(iframe)
iframe_html = driver.page_source
driver.switch_to.default_content()

# Embed iframe content with markers
page_html = page_html.replace('</body>',
    f'<!-- IFRAME_CONTENT_START -->{iframe_html}<!-- IFRAME_CONTENT_END --></body>')
```

## Results

### Before Fix
- **Inclusion Rate**: 2.8% (12 citations from 434 rows)
- **HTML File Size**: ~120KB (missing iframe content)

### After Fix (1 query tested)
- **Inclusion Rate**: 50.0% (11 citations from 22 results)  
- **HTML File Size**: ~609KB (includes iframe content)
- **Status**: ✅ Working correctly

## Key Insight

**The critical fix was iframe extraction, not just selector updates.** Citations are in a dynamically-loaded iframe that must be accessed with `driver.switch_to.frame()`.

## Status

- ✅ Google AI parser fix (99% → 8.7%)
- ✅ Bing iframe extraction working
- 🔄 Scraping 88 queries in progress (2-3 hours)
