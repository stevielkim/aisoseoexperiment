# SEO vs AISO: What Drives AI Search Engine Citations?

**A comprehensive analysis of what content features predict citation in AI-powered search engines (Google AI Overview, Bing Copilot, Perplexity AI)**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)]()
[![Status](https://img.shields.io/badge/Status-Active-success)]()

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Project Progression](#project-progression)
3. [Quick Start](#quick-start)
4. [Documentation](#documentation)
5. [Key Findings](#key-findings)
6. [Repository Structure](#repository-structure)
7. [Installation](#installation)
8. [Usage](#usage)

---

## 🎯 Overview

This project investigates **what makes content get cited by AI search engines**. As AI-powered search (Google AI Overview, Bing Copilot, Perplexity) becomes mainstream, understanding citation drivers is crucial for content creators and SEO professionals.

### Research Questions

1. **Do traditional SEO factors still matter?** (Page rank, word count, headings)
2. **What content features predict AI citation?** (Structure, depth, schema markup)
3. **Are there differences between engines?** (Google AI vs Bing AI vs Perplexity)
4. **What's the optimal content strategy?** (Actionable recommendations)

### Dataset

- **380 citations** across 73 queries
- **759 source pages** analyzed
- **60+ content features** extracted per page
- **3 AI search engines**: Perplexity (190 citations), Google AI (190 citations), Bing AI (deferred)

---

## 📈 Project Progression

This project evolved through several phases, each improving upon the last:

### Phase 1: Initial Data Collection (Sep 2025)
**Goal**: Collect raw HTML from AI search engines

**What Happened**:
- Built Selenium scrapers for 3 engines
- Successfully captured HTML files (1-2MB each)
- **Challenge**: CAPTCHA blocks, inconsistent results

**Output**: Raw HTML files stored in `data/raw/html/`

### Phase 2: Parser Development (Sep-Oct 2025)
**Goal**: Extract citations from AI Overview boxes

**What Happened**:
- **Initial Failure**: Google AI 97% failure rate, Bing AI 100% failure
- **Root Cause**: Parser selectors didn't match HTML structure
- **Solution**: Created debug scripts to inspect actual HTML
- **Breakthrough**: Found working selector (`div[data-initq]` for Google AI)

**Result**: Google AI citations improved from 0 → 190 ✅

### Phase 3: Feature Extraction (Oct-Nov 2025)
**Goal**: Extract 60+ content features from cited sources

**What Happened**:
- Fetched and parsed 759 source URLs
- Extracted: word count, headings, schema markup, domain info, content type
- **88.5% success rate** despite paywalls and JS-heavy sites

**Output**: `data/processed/source_features.csv` with 60+ features per URL

### Phase 4: Initial Analysis (Oct-Nov 2025)
**Goal**: Identify what predicts AI citations

**What Happened**:
- Basic correlation analysis
- Random Forest feature importance
- Initial visualizations

**Limitation**: Ad-hoc statistics, ~30% false positive risk due to no multiple comparison correction

### Phase 5: Statistical Hardening (Dec 2025) 🎯
**Goal**: Apply rigorous statistical methodology

**What Changed**:
- ✅ **FDR correction** for multiple comparisons (Benjamini-Hochberg) → <5% false discovery rate
- ✅ **Train-test splits** for model evaluation → detect overfitting
- ✅ **Confidence intervals** on all estimates → quantify uncertainty
- ✅ **Effect sizes** (Cohen's d, Cramér's V) → practical significance
- ✅ **Automatic test selection** (Pearson vs Spearman) → appropriate methods

**Result**: Publication-quality analysis with proper statistical rigor

### Phase 6: Code Refactoring (Dec 2025) 🏗️
**Goal**: Restructure to Python data science best practices

**What Changed**:
- **Modular code**: Functions extracted to `src/analysis/`, `src/visualization/`
- **Executable scripts**: Clean entry points in `scripts/`
- **Shared utilities**: Statistical functions in `src/analysis/statistical.py`
- **Documentation**: Comprehensive docstrings (NumPy style)
- **Installable package**: `pip install -e .`

**Result**: Testable, maintainable, professional code structure

**See**: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for navigation guide

---

## 🚀 Quick Start

```bash
# 1. Clone and install
git clone https://github.com/yourusername/geoseo_analysis.git
cd geoseo_analysis
pip install -r requirements.txt
pip install -e .

# 2. Run analyses (uses existing processed data)
python scripts/04_analyze_traditional_seo.py     # Traditional SEO factors
python scripts/07_analyze_content_features.py    # Content feature analysis

# 3. View results
open outputs/figures/traditional_seo_analysis.png
open outputs/figures/content_feature_analysis.png
```

---

## 📚 Documentation

### Core Documentation (Separate Files for Easy Navigation)

| Document | Description | Audience |
|----------|-------------|----------|
| **[METHODOLOGY.md](docs/METHODOLOGY.md)** | Statistical methods explained with external links | Beginner-Intermediate |
| **[RESULTS.md](docs/RESULTS.md)** | Detailed findings with embedded visualizations | All levels |
| **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** | Code organization & what goes where | Developers |
| **[USAGE_GUIDE.md](docs/USAGE_GUIDE.md)** | Step-by-step pipeline instructions | Users |
| **[DATA_QUALITY.md](docs/DATA_QUALITY.md)** | Data collection & quality assessment | Technical |

### Quick Navigation

- 🤔 **Want to understand the statistics?** → [METHODOLOGY.md](docs/METHODOLOGY.md)
- 📊 **Want to see the findings?** → [RESULTS.md](docs/RESULTS.md)
- 🚀 **Want to run it yourself?** → [USAGE_GUIDE.md](docs/USAGE_GUIDE.md)
- 🗺️ **Want to navigate the code?** → [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

---

## 🏆 Key Findings

### 1. Traditional SEO Still Matters ✅

**Google AI & Bing AI heavily favor top-ranking pages:**
- Rank 1-3 pages: **99.3%** inclusion rate (Google AI)
- Rank 4-10 pages: **98.8%** inclusion rate (Google AI)
- **χ² = 763.08, p < 0.0001, Cramér's V = 0.865** (large effect)

**Top predictive features (Random Forest):**
1. **H2 Count** (importance: 0.303)
2. **H1 Count** (importance: 0.279)
3. **Page Rank** (importance: 0.192)

### 2. Content Structure is Critical 📋

**Logistic Regression Odds Ratios** (how much each feature increases citation odds):
- **Word Count**: 14.94x higher odds per unit increase
- **H1 Count**: 2.23x higher odds
- **Image Count**: 1.71x higher odds

**Model Performance**:
- Random Forest: 87.7% test accuracy (good generalization)
- Logistic Regression: 92.6% test accuracy

### 3. Domain Authority Advantage 🎓

**Most cited domains:**
1. bing.com (159 citations)
2. reddit.com (44 citations)
3. mayoclinic.org (31 citations)
4. healthline.com (25 citations)

**Domain type distribution:**
- Commercial (.com): **67.9%**
- Organization (.org): **16.9%**
- Educational (.edu): **4.2%**
- Government (.gov): **0.9%**

**Takeaway**: Authoritative health, education, and government sites are frequently cited

### 4. Citation Order Patterns 📊

**Early citations (positions 1-3) have distinct features:**
- Position 1: **19.9%** of all citations
- Position 2: **5.7%**
- Position 3: **4.2%**

**Features predicting early citation (Random Forest):**
- word_count (0.171 importance)
- paragraph_count (0.119)
- external_link_count (0.118)

### 5. Data Quality Insights ⚠️

**Engine reliability:**
- ✅ **Perplexity**: 96.7% inclusion rate (reliable, but high)
- ⚠️ **Google AI**: 99.0% inclusion rate (suspiciously high - may capture beyond AI Overview)
- ✅ **Bing AI**: 14.3% inclusion rate (expected range)

**See [DATA_QUALITY.md](docs/DATA_QUALITY.md) for full assessment**

---

## 📁 Repository Structure

```
geoseo_analysis/
├── README.md                    # This file
├── PROJECT_STRUCTURE.md         # Detailed navigation guide
├── requirements.txt             # Dependencies
├── setup.py                     # Package config
│
├── data/                        # Data pipeline (gitignored)
│   ├── raw/html/                # Scraped HTML (3 engines)
│   ├── interim/                 # Intermediate data
│   └── processed/               # Final datasets
│       ├── ai_serp_analysis.csv       # Main citation data
│       ├── citations_valid.csv        # Cleaned citations
│       └── source_features.csv        # 60+ features per source
│
├── src/                         # Modular source code (NEW - refactored)
│   ├── analysis/                # Analysis functions
│   │   ├── statistical.py       # FDR, CI, effect sizes
│   │   ├── content_features.py  # Content analysis
│   │   └── traditional_seo.py   # SEO analysis
│   ├── visualization/           # Plotting functions
│   │   └── dashboards.py        # Dashboard creation
│   └── utils/                   # Utilities
│       └── logging.py
│
├── scripts/                     # Executable scripts (NEW - refactored)
│   ├── 04_analyze_traditional_seo.py    # ✅ Run this
│   └── 07_analyze_content_features.py   # ✅ Run this
│
├── outputs/                     # Generated outputs
│   ├── figures/                 # Visualizations
│   │   ├── traditional_seo_analysis.png    # Latest dashboard
│   │   └── content_feature_analysis.png    # Latest dashboard
│   ├── reports/                 # Analysis reports
│   └── models/                  # Saved models
│
├── docs/                        # Documentation
│   ├── METHODOLOGY.md           # Statistical methods (NEW)
│   ├── RESULTS.md               # Detailed findings (NEW)
│   ├── USAGE_GUIDE.md           # Usage instructions
│   └── DATA_QUALITY.md          # Data quality report
│
├── config/                      # Configuration
│   └── analysis_config.yaml     # Analysis parameters
│
└── analyzegeo/                  # Legacy pipeline scripts
    ├── scrape_geo.py            # Data collection (not refactored)
    ├── parse_citations.py       # Citation extraction (not refactored)
    ├── fetch_source_features.py # Feature extraction (not refactored)
    └── legacy/                  # Old analysis scripts (gitignored)
```

---

## 💻 Installation

### Requirements

- Python 3.9+
- pip package manager
- Virtual environment (recommended)

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

### Core Dependencies

- `pandas` - Data manipulation
- `scikit-learn` - Machine learning
- `scipy`, `statsmodels` - Statistical tests
- `matplotlib`, `seaborn` - Visualizations
- `selenium`, `beautifulsoup4` - Web scraping

See [requirements.txt](requirements.txt) for complete list.

---

## 🔧 Usage

### Running Analyses (Recommended)

```bash
# Traditional SEO analysis (Google AI + Bing AI)
python scripts/04_analyze_traditional_seo.py

# Content feature analysis (all engines)
python scripts/07_analyze_content_features.py
```

### Full Data Pipeline (Advanced)

If you want to collect fresh data:

```bash
cd analyzegeo/

# 1. Scrape search results
python scrape_geo.py

# 2. Parse citations
python parse_citations.py

# 3. Extract features
python fetch_source_features.py

# 4. Run analyses
cd ..
python scripts/04_analyze_traditional_seo.py
python scripts/07_analyze_content_features.py
```

**See [USAGE_GUIDE.md](docs/USAGE_GUIDE.md) for detailed instructions**

---

## 🤝 Contributing

Contributions welcome! The project needs:

**High Priority:**
- Refactor remaining 2 analysis scripts (ai_citations, combined_insights)
- Complete documentation suite (METHODOLOGY.md, RESULTS.md)
- Add unit tests for src/ modules

**Medium Priority:**
- Refactor data pipeline scripts (scrape, parse, extract)
- Create interactive dashboards (Plotly/Dash)
- Expand query dataset

**Please:**
1. Fork the repository
2. Create a feature branch
3. Write clear commit messages
4. Add docstrings to new functions
5. Open a Pull Request

---

## 📧 Contact

- **Author**: Stephanie Kim
- **Project**: [github.com/yourusername/geoseo_analysis](https://github.com/yourusername/geoseo_analysis)

---

## 🙏 Acknowledgments

- **AI Search Engines**: Google AI Overview, Bing Copilot, Perplexity AI
- **Libraries**: Selenium, BeautifulSoup, scikit-learn, statsmodels, matplotlib
- **Development**: Claude Code for refactoring assistance

---

**⭐ Star this repo if you find it useful!**

**Last Updated**: December 2025
**Project Status**: Active - Documentation Phase
