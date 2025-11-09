# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an SEO vs AISO (AI Search Optimization) research project that compares traditional SEO search results with AI-generated answers from Google AI Overviews, Perplexity, and Bing Copilot. The goal is to discover whether the same content is prioritized by traditional search engines vs AI search engines/overviews.

## Project Structure

The main working code is located in the `analyzegeo/` directory. All development should be done within this subdirectory unless specifically working with root-level configuration files.

```
├── analyzegeo/              # Main codebase directory
│   ├── scrape_geo.py       # Data collection with Selenium
│   ├── parse_geo.py        # HTML parsing and feature extraction
│   ├── analyze_geo.py      # Statistical analysis and visualization
│   ├── debug_*.py          # Engine-specific debugging scripts
│   └── CLAUDE.md           # Detailed development guidance
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

## Development Workflow

### Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

### Core Pipeline
```bash
# Navigate to main code directory
cd analyzegeo/

# 1. Scrape search results (requires proper query file path)
python scrape_geo.py

# 2. Parse HTML files to extract features
python parse_geo.py

# 3. Generate analysis and visualizations
python analyze_geo.py
```

## Key Features

- **Multi-engine scraping**: Perplexity, Google AI Overviews, Bing Copilot
- **SEO feature extraction**: Word count, headings, meta descriptions, schema markup
- **AI citation analysis**: Determines which pages are included in AI-generated answers
- **Statistical modeling**: Logistic regression to identify inclusion drivers
- **Visualization dashboard**: Comprehensive analysis plots

## Working Directory

**Important**: The primary codebase is in the `analyzegeo/` directory. When working on this project, always `cd analyzegeo/` first or reference files with the full path. The `analyzegeo/CLAUDE.md` file contains detailed architectural information and specific development commands for the core functionality.

## Dependencies

The project uses Python with web scraping (Selenium, BeautifulSoup), data analysis (pandas, scikit-learn), and visualization (matplotlib, seaborn) libraries. See `requirements.txt` for the complete list.