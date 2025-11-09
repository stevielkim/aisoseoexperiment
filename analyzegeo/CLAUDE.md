# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project comparing traditional SEO search results with AI-generated answers from Google AI Overviews, Perplexity, and Bing Copilot. The goal is to discover if the same content is prioritized by traditional search engines vs AI search engines/overviews to understand if content needs to be rewritten to optimize for AI Search Optimization (AISO).

## Core Architecture

The codebase follows a three-stage pipeline:

1. **Data Collection (`scrape_geo.py`)**: Uses Selenium with undetected Chrome driver to scrape search results from three AI search engines:
   - Perplexity.ai - Direct search queries
   - Google AI - Search with AI Overview extraction
   - Bing Copilot - SERP with Copilot tab interaction

2. **Data Parsing (`parse_geo.py`)**: Processes the scraped HTML files to extract SEO and structural features:
   - Basic SEO elements (title, meta description, word count)
   - Structural features (headings, lists, tables, images)
   - Schema markup detection (FAQ, HowTo, Article)
   - AI overview content and citation analysis
   - Outputs consolidated CSV for analysis

3. **Analysis (`analyze_geo.py`)**: Performs statistical analysis on the parsed data:
   - Logistic regression to identify inclusion drivers
   - Correlation analysis between page rank and inclusion
   - Schema impact analysis with chi-square tests
   - Generates visualization dashboard with matplotlib/seaborn

## Common Development Commands

### Running the Full Pipeline

#### Option 1: Web Scraping (Legacy)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Scrape data (requires query file at correct path)
python scrape_geo.py

# 3. Parse HTML files to CSV
python parse_geo.py

# 4. Generate analysis and plots
python analyze_geo.py
```

#### Option 2: API Integration (Recommended for Perplexity)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up Perplexity API key
export PERPLEXITY_API_KEY='your-api-key-here'

# 3. Collect Perplexity data via API
python scrape_perplexity_api.py

# 4. Scrape Google AI and Bing AI (still using web scraping)
python scrape_geo.py  # Will skip Perplexity if API data exists

# 5. Parse all data to CSV
python parse_geo.py

# 6. Generate analysis and plots
python analyze_geo.py
```

### Testing and Debugging
```bash
# Test single file parsing
python test_single.py

# Debug specific engine parsing
python debug_detailed.py
python debug_google_ai.py
python debug_perplexity.py

# Run with pytest (if tests exist)
pytest

# Code formatting
black *.py
flake8 *.py
```

## Key Dependencies

- **Web Scraping**: `selenium`, `undetected-chromedriver`, `webdriver-manager`
- **HTML Parsing**: `beautifulsoup4`, `lxml`
- **Data Analysis**: `pandas`, `numpy`, `scikit-learn`
- **Visualization**: `matplotlib`, `seaborn`
- **Statistical Analysis**: `scipy`

## Important File Locations

- Query file path in `scrape_geo.py`: `/Users/stephaniekim/development/seoaso/seo_vs_aiso_analysis/queries/seo_aso_prompts.txt`
- Output directories: `perplexity_search_results_html/`, `google_ai_search_results_html/`, `bing_ai_search_results_html/`
- Analysis output: `ai_serp_analysis.csv`, `plots/ai_inclusion_dashboard.png`

## Data Structure

The main CSV output (`ai_serp_analysis.csv`) contains:
- Basic SEO metrics (Word Count, H1/H2/H3 counts, MetaDesc Length)
- Structural features (List Count, Table Count, Image Count, etc.)
- Schema flags (Has FAQ Schema, Has HowTo Schema, Has Article Schema)
- Inclusion analysis (Included flag, Citation Order, Citation Paragraph)
- Engine-specific data for comparison

## Development Notes

- The scraping uses `undetected-chromedriver` to avoid bot detection
- HTML parsing includes engine-specific selectors for AI overview extraction
- Analysis pipeline includes extensive logging for debugging
- All URL normalization removes protocols and www prefixes for comparison
- Debug scripts help troubleshoot engine-specific parsing issues