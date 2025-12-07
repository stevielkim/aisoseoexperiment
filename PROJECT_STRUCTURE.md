# Project Structure Guide

## Overview

This project follows **Python data science best practices** (cookiecutter-data-science layout) with a clear separation between:
- **Data** (raw → interim → processed)
- **Source code** (modular, importable functions)
- **Executable scripts** (entry points that call source code)
- **Outputs** (generated figures, reports, models)
- **Documentation**

---

## Current Structure (After Refactoring)

```
geoseo_analysis/
├── README.md                      # Main project documentation
├── PROJECT_STRUCTURE.md           # This file - navigation guide
├── setup.py                       # Package installation config
├── requirements.txt               # Python dependencies
│
├── data/                          # Data pipeline (gitignored)
│   ├── raw/                       # Original, immutable data
│   │   └── html/                  # Scraped HTML files (3 engines)
│   ├── interim/                   # Intermediate data
│   │   └── source_features_raw.csv
│   └── processed/                 # Final, canonical datasets
│       ├── ai_serp_analysis.csv
│       ├── citations_valid.csv
│       └── source_features.csv
│
├── src/                           # Modular source code (importable)
│   ├── __init__.py
│   ├── analysis/                  # Analysis modules
│   │   ├── __init__.py
│   │   ├── statistical.py         # Statistical utilities (FDR, CI, effect sizes)
│   │   ├── content_features.py    # Content feature analysis
│   │   └── traditional_seo.py     # Traditional SEO analysis
│   ├── visualization/             # Plotting functions
│   │   ├── __init__.py
│   │   └── dashboards.py          # Dashboard creation
│   └── utils/                     # Utilities
│       ├── __init__.py
│       └── logging.py             # Logging configuration
│
├── scripts/                       # Executable entry points
│   ├── 04_analyze_traditional_seo.py    # ✅ REFACTORED
│   └── 07_analyze_content_features.py   # ✅ REFACTORED
│
├── outputs/                       # Generated outputs
│   ├── figures/                   # Plots and visualizations
│   │   ├── traditional_seo_analysis.png
│   │   └── content_feature_analysis.png
│   ├── reports/                   # Analysis reports
│   └── models/                    # Saved model artifacts
│
├── docs/                          # Documentation
│   ├── METHODOLOGY.md
│   ├── USAGE_GUIDE.md
│   ├── DATA_QUALITY.md
│   └── PERPLEXITY_API_SETUP.md
│
├── config/                        # Configuration files
│   └── analysis_config.yaml       # Analysis parameters, paths, selectors
│
├── analyzegeo/                    # Legacy pipeline scripts (NOT refactored yet)
│   ├── scrape_geo.py              # Data collection (Selenium)
│   ├── parse_citations.py         # Citation extraction
│   ├── parse_geo.py               # HTML parsing
│   ├── fetch_source_features.py   # Feature extraction
│   ├── debug_*.py                 # Debugging utilities
│   └── legacy/                    # Old analysis scripts (superseded by scripts/)
│       ├── analyze_content_features.py    # OLD - use scripts/07_* instead
│       ├── analyze_traditional_seo.py     # OLD - use scripts/04_* instead
│       ├── analyze_ai_citations.py        # OLD - not yet refactored
│       └── analyze_combined_insights.py   # OLD - not yet refactored
│
└── queries/
    └── seo_aso_prompts.txt        # Query list for scraping
```

---

## What Goes Where?

### ✅ USE THESE (Refactored, Best Practices)

**For Running Analysis:**
```bash
# Traditional SEO analysis (Google AI + Bing AI)
python scripts/04_analyze_traditional_seo.py

# Content feature analysis (all engines)
python scripts/07_analyze_content_features.py
```

**For Importing Functions:**
```python
from src.analysis.content_features import analyze_citation_patterns
from src.analysis.statistical import test_correlations_with_fdr
from src.visualization.dashboards import create_content_feature_dashboard
```

### ⚠️ LEGACY (Old, Not Refactored Yet)

**Data Pipeline (still in analyzegeo/):**
```bash
cd analyzegeo/

# Step 1: Scrape search results
python scrape_geo.py

# Step 2: Parse citations
python parse_citations.py

# Step 3: Extract features
python fetch_source_features.py
```

**Old Analysis Scripts (in analyzegeo/legacy/):**
- `analyze_content_features.py` → **Replaced by** `scripts/07_analyze_content_features.py`
- `analyze_traditional_seo.py` → **Replaced by** `scripts/04_analyze_traditional_seo.py`
- `analyze_ai_citations.py` → **Not yet refactored** (Perplexity-only analysis)
- `analyze_combined_insights.py` → **Not yet refactored** (cross-engine synthesis)

---

## Refactoring Status

| Script | Status | Location | Notes |
|--------|--------|----------|-------|
| **Data Pipeline** | | | |
| `scrape_geo.py` | ⏳ Not refactored | `analyzegeo/` | Works, but not modular yet |
| `parse_citations.py` | ⏳ Not refactored | `analyzegeo/` | Works, but not modular yet |
| `fetch_source_features.py` | ⏳ Not refactored | `analyzegeo/` | Works, but not modular yet |
| **Analysis Scripts** | | | |
| `analyze_content_features.py` | ✅ Refactored | `scripts/07_*` + `src/analysis/content_features.py` | Uses modular functions |
| `analyze_traditional_seo.py` | ✅ Refactored | `scripts/04_*` + `src/analysis/traditional_seo.py` | Uses modular functions |
| `analyze_ai_citations.py` | ⏳ Not refactored | `analyzegeo/legacy/` | Perplexity-specific analysis |
| `analyze_combined_insights.py` | ⏳ Not refactored | `analyzegeo/legacy/` | Cross-engine synthesis |

---

## Quick Start

### Running the Full Pipeline

```bash
# 1. Scrape data (if needed)
cd analyzegeo && python scrape_geo.py
cd ..

# 2. Parse citations (if needed)
cd analyzegeo && python parse_citations.py
cd ..

# 3. Extract features (if needed)
cd analyzegeo && python fetch_source_features.py
cd ..

# 4. Run analyses (refactored versions)
python scripts/04_analyze_traditional_seo.py     # Traditional SEO
python scripts/07_analyze_content_features.py    # Content features

# 5. View results
open outputs/figures/traditional_seo_analysis.png
open outputs/figures/content_feature_analysis.png
```

### Installing the Package

```bash
# Install in editable mode (recommended for development)
pip install -e .

# Now you can import from anywhere
python
>>> from src.analysis.statistical import audit_data_quality
>>> from src.visualization.dashboards import create_content_feature_dashboard
```

---

## Benefits of New Structure

**Modular Code:**
- Functions can be imported and reused
- Easier to test individual components
- No code duplication

**Statistical Rigor:**
- FDR correction for multiple comparisons
- Train-test splits for model evaluation
- Confidence intervals on all estimates
- Effect size calculations (Cohen's d, Cramér's V)

**Professional Organization:**
- Follows industry-standard layout
- Clear separation: data / code / outputs / docs
- Pip-installable package
- Comprehensive docstrings

---

## Next Steps

1. **Complete refactoring** of remaining 2 analysis scripts:
   - `analyze_ai_citations.py` → `scripts/05_analyze_ai_citations.py`
   - `analyze_combined_insights.py` → `scripts/06_analyze_combined.py`

2. **Refactor data pipeline** scripts:
   - Move functions from `scrape_geo.py` → `src/data/scrape.py`
   - Move functions from `parse_citations.py` → `src/data/parse.py`
   - Move functions from `fetch_source_features.py` → `src/data/features.py`
   - Create executable wrappers in `scripts/01_scrape.py`, `scripts/02_parse.py`, `scripts/03_extract_features.py`

3. **Create comprehensive documentation**:
   - Enhanced METHODOLOGY.md with statistical concepts
   - RESULTS.md with embedded visualizations
   - OPTIMIZATION_GUIDE.md with actionable recommendations

---

## Questions?

- **"Which script should I run?"** → Use `scripts/*` (refactored versions)
- **"Why are there scripts in analyzegeo/?"** → Those are legacy pipeline scripts not yet refactored
- **"Can I delete analyzegeo/legacy/?"** → No, keep as reference until all scripts are refactored
- **"How do I import functions?"** → `from src.analysis.module_name import function_name`
