# SEO vs AISO Analysis Project

A comprehensive research project comparing traditional SEO (Search Engine Optimization) with AISO (AI Search Optimization) across Google AI Overviews, Bing Copilot, and Perplexity citations.

## 🎯 Project Overview

This project investigates whether content optimized for traditional search engines performs similarly in AI-powered search systems. Through extensive data collection and analysis, we discovered that **traditional search engines and AI citation systems operate fundamentally differently**, requiring separate optimization strategies.

### Key Research Question
> *Do traditional SEO factors predict inclusion in AI-generated search responses, or do AI systems prioritize different content characteristics?*

### Core Findings
- **Traditional search engines** (Google AI, Bing AI) still rely heavily on PageRank and traditional SEO factors
- **AI citation systems** (Perplexity) prioritize content authority, trustworthiness, and citation-worthy factual information
- **No universal optimization strategy** works across all systems - dual-track approach required

## 📊 Dual-Track Analysis Methodology

Based on our analysis, we developed a **dual-track approach** that treats different engine types separately:

### Track 1: Traditional SEO Analysis
- **Engines**: Google AI + Bing AI
- **Focus**: Traditional search ranking factors in AI Overview inclusion
- **Methodology**: Standard SEO analysis adapted for AI enhancements

### Track 2: AI Citation Analysis
- **Engine**: Perplexity
- **Focus**: Content characteristics that earn AI citations
- **Methodology**: Citation pattern analysis and content quality assessment

## 🏗️ Project Structure

```
geoseo_analysis/
├── README.md                           # This documentation
├── analyzegeo/                         # Main analysis directory
│   ├── CLAUDE.md                      # AI assistant guidance
│   ├── scrape_geo.py                  # Web scraping pipeline
│   ├── parse_geo.py                   # HTML parsing & feature extraction
│   ├── analyze_traditional_seo.py     # Traditional SEO analysis
│   ├── analyze_ai_citations.py        # AI citation analysis
│   ├── analyze_combined_insights.py   # Combined strategic insights
│   ├── scrape_perplexity_api.py      # Perplexity API integration
│   └── plots/                         # Generated visualizations
├── queries/                           # Query datasets
│   └── seo_aso_prompts.txt           # 88 diverse search queries
├── compare_results/                   # Comparison utilities
└── [engine]_search_results_html/     # Raw HTML data
    ├── google_ai_search_results_html/
    ├── bing_ai_search_results_html/
    └── perplexity_search_results_html/
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Virtual environment recommended
- Chrome browser (for Selenium)

### Installation
```bash
# Clone and navigate to project
cd /path/to/geoseo_analysis/analyzegeo

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage
```bash
# Run complete analysis pipeline
python scrape_geo.py          # 1. Scrape search results
python parse_geo.py           # 2. Extract features from HTML
python analyze_traditional_seo.py    # 3a. Traditional SEO analysis
python analyze_ai_citations.py      # 3b. AI citation analysis
python analyze_combined_insights.py # 4. Strategic synthesis
```

## 📈 Analysis Scripts

### 1. `analyze_traditional_seo.py`
**Focus**: Google AI + Bing AI traditional search analysis

**Key Features**:
- Traditional SEO factor correlation analysis
- Page rank vs inclusion modeling
- Predictive modeling (Random Forest, Logistic Regression)
- Engine-specific performance comparison

**Outputs**:
- `plots/traditional_seo_analysis.png` - Comprehensive dashboard
- Traditional SEO insights and recommendations

**Usage**:
```bash
python analyze_traditional_seo.py
```

### 2. `analyze_ai_citations.py`
**Focus**: Perplexity AI citation pattern analysis

**Key Features**:
- Citation order vs inclusion analysis
- Content quality factor identification
- Query type preference analysis
- Citation source authority assessment

**Outputs**:
- `plots/ai_citation_analysis.png` - AI citation dashboard
- Citation optimization recommendations

**Usage**:
```bash
python analyze_ai_citations.py
```

### 3. `analyze_combined_insights.py`
**Focus**: Strategic synthesis and cross-system insights

**Key Features**:
- Cross-engine query performance comparison
- Universal optimization principle identification
- Dual-track strategy recommendations
- Data quality assessment across systems

**Outputs**:
- `plots/combined_insights_analysis.png` - Strategic overview
- Comprehensive optimization strategy

**Usage**:
```bash
python analyze_combined_insights.py
```

## 📊 Key Metrics & Findings

### Traditional SEO Track (Google AI + Bing AI)
| Metric | Google AI | Bing AI | Combined |
|--------|-----------|---------|----------|
| Total Results | 586 | 69 | 655 |
| Inclusion Rate | 0.3% | 8.7% | 1.2% |
| Top Factor | Page Rank | Page Rank | Page Rank |
| Optimal Length | ~1,855 words | ~1,855 words | ~1,855 words |

### AI Citation Track (Perplexity)
| Metric | Value | Notes |
|--------|-------|-------|
| Total Citations | 304 | - |
| Inclusion Rate | 97.0% | ⚠️ Data quality issue |
| Top Factor | AI Overview Length | 0.324 importance |
| Optimal Length | ~377 words | Much shorter than traditional |

### Data Quality Issues Identified
1. **Perplexity**: 97.0% inclusion rate indicates web scraping parsing errors
2. **Google AI**: 0.3% inclusion rate suggests AI Overview selector issues
3. **Bing AI**: Only engine with reasonable inclusion patterns (8.7%)

## 🔧 Technical Details

### Data Collection Pipeline
1. **Web Scraping** (`scrape_geo.py`)
   - Uses Selenium with undetected Chrome driver
   - Scrapes Google AI Overviews, Bing Copilot responses, Perplexity citations
   - Handles dynamic content loading and bot detection

2. **HTML Parsing** (`parse_geo.py`)
   - Extracts 30+ SEO and content quality features
   - Engine-specific selectors for AI content detection
   - Normalizes URLs and handles missing data

3. **Feature Engineering**
   - Traditional SEO metrics (word count, headings, meta descriptions)
   - Structural features (lists, tables, images)
   - Schema markup detection (FAQ, HowTo, Article)
   - AI-specific metrics (overview length, citation order)

### Machine Learning Models
- **Random Forest Classifier**: Feature importance and inclusion prediction
- **Logistic Regression**: Linear factor analysis
- **Statistical Tests**: Chi-square, Mann-Whitney U for significance testing

## 📋 Query Dataset

The project analyzes **88 diverse queries** across 6 categories:

| Category | Count | Examples |
|----------|-------|----------|
| How-to | 38 | "how to write a resume", "how to start a business" |
| Informational | 18 | "what is blockchain", "symptoms of vitamin D deficiency" |
| Best-of | 20 | "best programming languages 2025", "best cloud storage" |
| Comparison | 21 | "iPhone vs Samsung", "Python vs JavaScript" |
| Benefits | 3 | "benefits of remote work", "benefits of meditation" |
| Other | 21 | Various specific queries |

## ⚠️ Known Issues & Limitations

### Data Quality Concerns
1. **Perplexity Web Scraping**: 97% inclusion rate indicates parsing errors
   - **Solution**: Implement Perplexity API integration (`scrape_perplexity_api.py`)

2. **Google AI Selectors**: 0.3% inclusion rate suggests selector failures
   - **Solution**: Update CSS selectors for current AI Overview structure

3. **Query Overlap**: No common queries between traditional and AI systems
   - **Impact**: Limits cross-system comparison capabilities

### Technical Limitations
- Web scraping subject to site changes and bot detection
- Large dataset analysis scripts may timeout (>2 minutes)
- Some visualizations require non-interactive matplotlib backend

## 🔮 Future Improvements

### High Priority
1. **Perplexity API Integration**
   - Replace web scraping with official API
   - Get accurate citation data and inclusion metrics
   - Script already created: `scrape_perplexity_api.py`

2. **Google AI Selector Updates**
   - Debug and fix AI Overview detection
   - Improve inclusion rate accuracy
   - Test across different query types

### Medium Priority
3. **Query Dataset Expansion**
   - Ensure query overlap between systems
   - Add 50+ more diverse queries
   - Enable better cross-system comparisons

4. **Longitudinal Analysis**
   - Track changes over time
   - Monitor algorithm updates
   - Identify trending patterns

### Low Priority
5. **Additional Engines**
   - Claude.ai search integration
   - ChatGPT search analysis
   - Expand AI system coverage

## 📚 Research Implications

### For SEO Professionals
- **Traditional SEO remains relevant** for AI Overview inclusion
- **Page rank still matters** in AI-enhanced search
- **Content optimization should target both systems** separately

### For Content Creators
- **Dual-track content strategy** recommended
- **Authority and trustworthiness** critical for AI citations
- **Content length optimization** varies by system type

### For Researchers
- **Separate methodologies required** for different engine types
- **Data quality is critical** in AI search research
- **Cross-system studies need careful design** to ensure validity

## 🤝 Contributing

This project welcomes contributions in several areas:

### Data Collection
- Improve web scraping reliability
- Add new search engines
- Expand query datasets

### Analysis
- Enhance machine learning models
- Add new feature engineering
- Improve statistical analysis

### Documentation
- Add use case examples
- Improve installation guides
- Create video tutorials

## 📄 License

This project is for research and educational purposes. Please respect the terms of service of all search engines and APIs used.

## 🙏 Acknowledgments

- **Search Engines**: Google, Microsoft Bing, Perplexity AI for providing the data
- **Open Source Libraries**: Selenium, BeautifulSoup, pandas, scikit-learn, matplotlib
- **Research Community**: SEO and AI research communities for inspiration and methodologies

---

**Last Updated**: November 2025
**Project Status**: Active Development
**Research Phase**: Data Quality Improvement & API Integration
