# Scraping Plan - Technology Queries (December 2025)

## Overview

Adding 50 new technology-focused queries to capture timeseries data for Perplexity and Google AI Overview.

## Changes Made

### 1. Query File Updated
**File**: `queries/seo_aso_prompts.txt`
- **Old count**: 88 queries
- **New count**: 138 queries (added 50 technology queries)
- **New queries start at line 89**

### 2. New Technology Queries Added (50 total)

**Categories**:
- **Hardware**: PC building, laptops, monitors, keyboards, headphones
- **Software**: Operating systems, apps, video editing, password managers
- **Networking**: Wi-Fi, VPN, 5G, routers, Internet of Things
- **Mobile**: iPhone vs Android, smartphones, backup strategies
- **Emerging Tech**: AI, metaverse, quantum computing, edge computing
- **Troubleshooting**: Speed optimization, file recovery, setup guides

**Query Types**:
- **Informational (how/what)**: 25 queries (50%)
  - "how to build a PC", "what is cloud computing", "how does Wi-Fi work"
- **Transactional (best/comparison)**: 25 queries (50%)
  - "best laptop for college students", "iPhone 16 vs iPhone 15", "best gaming laptops 2025"

### 3. Scraper Updated for Timeseries Data
**File**: `analyzegeo/scrape_geo.py`

**Key changes**:
1. **Timestamped directories**: Output now goes to `perplexity_search_results_html_YYYYMMDD_HHMMSS/`
2. **Preserves old data**: Won't overwrite previous scraping runs
3. **Skips Bing**: Bot detection issues prevent reliable Bing data collection
4. **Better progress tracking**: Shows `[N/138]` progress and timing info

**Output directories** (example):
```
data/raw/html/
├── perplexity_search_results_html_20251223_143045/  # New run
├── google_ai_search_results_html_20251223_143045/   # New run
├── perplexity_search_results_html/                  # Old data (preserved)
└── google_ai_search_results_html/                   # Old data (preserved)
```

## Running the Scraper

```bash
cd analyzegeo/
python scrape_geo.py
```

**Expected runtime**:
- 138 queries × 2 engines = 276 page loads
- ~5-10 seconds per page = ~23-46 minutes total
- With delays: ~45-60 minutes

## After Scraping

### 1. Update Parser
You'll need to update `parse_geo.py` to process the new timestamped directories:

```python
# Option 1: Update directory paths to latest timestamp
PERPLEXITY_DIR = "data/raw/html/perplexity_search_results_html_20251223_143045"
GOOGLE_AI_DIR = "data/raw/html/google_ai_search_results_html_20251223_143045"

# Option 2: Parse all directories for timeseries analysis
# Loop through all timestamped directories
```

### 2. Compare Results
- Compare inclusion rates across time periods
- Analyze if new technology queries show different patterns
- Check for changes in Google AI Overview behavior over time

## Expected Findings

**Hypotheses to test**:
1. Do technology queries have different inclusion rates than health/finance queries?
2. Are "how to" tech queries cited more than "best" tech queries?
3. Has Google AI Overview behavior changed over time?
4. Do newer tech topics (AI, metaverse) get cited differently than established topics?

## Bing Status

**Current status**: ❌ Postponed
- **Issue**: Bot detection blocking 98.6% of queries (70/71 failed)
- **Symptom**: "No results" error pages captured instead of actual responses
- **Attempted fixes**: Enhanced bot evasion, iframe extraction (working for 1 query)
- **Next steps**: Manual collection or API access required

## Files Modified

1. `/queries/seo_aso_prompts.txt` - Added 50 technology queries
2. `/analyzegeo/scrape_geo.py` - Timestamped output + Bing skipped
