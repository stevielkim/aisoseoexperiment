# SEO vs AISO Analysis - Usage Guide

This guide provides step-by-step instructions for running the complete SEO vs AISO analysis pipeline.

## 🚀 Quick Start

### Prerequisites Check
```bash
# Verify Python version (3.11+ required)
python --version

# Verify Git (for version control)
git --version

# Verify Chrome browser is installed
# The scraping pipeline requires Chrome for Selenium
```

### Environment Setup
```bash
# Navigate to the analyzegeo directory
cd /path/to/geoseo_analysis/analyzegeo

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Verify virtual environment is active (should show .venv in path)
which python

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import selenium, pandas, matplotlib; print('All dependencies installed successfully')"
```

## 📊 Complete Analysis Pipeline

### Step 1: Data Collection
```bash
# Scrape search results from all engines (Google AI, Bing AI, Perplexity)
# This will take 30-60 minutes depending on query count
python scrape_geo.py

# Expected output:
# - Creates HTML files in [engine]_search_results_html/ directories
# - Progress updates for each query processed
# - Final summary of successful scrapes
```

**What this does:**
- Scrapes Google AI Overviews, Bing Copilot responses, and Perplexity citations
- Handles dynamic content loading and bot detection
- Saves raw HTML files for later parsing

### Step 2: Feature Extraction
```bash
# Parse HTML files and extract SEO features
python parse_geo.py

# Expected output:
# - ai_serp_analysis.csv with all extracted features
# - Progress updates for each HTML file processed
# - Summary statistics of extracted data
```

**What this does:**
- Extracts 30+ SEO and content quality features from HTML
- Detects AI Overview content and citations
- Creates consolidated dataset for analysis

### Step 3a: Traditional SEO Analysis
```bash
# Analyze Google AI + Bing AI (traditional search engines)
python analyze_traditional_seo.py

# Expected output:
# - plots/traditional_seo_analysis.png
# - Console output with key findings and statistics
```

**Key insights provided:**
- Traditional SEO factor correlations
- Page rank vs AI Overview inclusion patterns
- Predictive model performance
- Engine-specific differences

### Step 3b: AI Citation Analysis
```bash
# Analyze Perplexity (AI citation system)
python analyze_ai_citations.py

# Expected output:
# - plots/ai_citation_analysis.png
# - Citation pattern analysis and insights
```

**Key insights provided:**
- Citation order vs inclusion patterns
- Content quality factors for AI citations
- Source authority analysis
- Query type preferences

### Step 4: Combined Strategic Analysis
```bash
# Generate strategic insights combining both tracks
python analyze_combined_insights.py

# Expected output:
# - plots/combined_insights_analysis.png
# - Strategic recommendations for dual-track optimization
```

**Key insights provided:**
- Cross-system optimization strategies
- Universal principles identification
- Data quality assessment
- Strategic recommendations

## 🔧 Individual Script Usage

### Data Collection Scripts

#### `scrape_geo.py` - Web Scraping Pipeline
```bash
# Basic usage
python scrape_geo.py

# The script will:
# 1. Load queries from ../queries/seo_aso_prompts.txt
# 2. Scrape each engine sequentially
# 3. Save HTML files to respective directories
# 4. Display progress and completion statistics
```

**Configuration options** (edit in script if needed):
- `QUERIES_FILE`: Path to query file
- `OUTPUT_DIRS`: HTML output directories
- `SELENIUM_OPTIONS`: Browser configuration

#### `scrape_perplexity_api.py` - API Integration (Future)
```bash
# Set up API key first
export PERPLEXITY_API_KEY='your-api-key-here'

# Run API scraping
python scrape_perplexity_api.py

# This will:
# 1. Use official Perplexity API instead of web scraping
# 2. Get accurate citation data
# 3. Solve the 97% inclusion rate data quality issue
```

### Data Processing Scripts

#### `parse_geo.py` - Feature Extraction
```bash
# Basic usage
python parse_geo.py

# Advanced usage with specific engines
# (Edit ENGINE_DIRS in script to focus on specific engines)
python parse_geo.py

# Debug mode for single files
python debug_detailed.py        # Debug all engines
python debug_perplexity.py     # Debug Perplexity specifically
python debug_google_ai.py      # Debug Google AI specifically
```

**Features extracted:**
- Basic SEO: Word count, headings, meta descriptions
- Structure: Lists, tables, images, paragraphs
- Schema: FAQ, HowTo, Article markup detection
- AI-specific: Overview length, citation order, inclusion status

### Analysis Scripts

#### `analyze_traditional_seo.py` - Traditional SEO Analysis
```bash
# Run analysis
python analyze_traditional_seo.py

# Output interpretation:
# - Green bars: Good performance
# - Red/orange bars: Data quality issues
# - Correlation values: >0.1 meaningful, >0.3 strong
# - Model accuracy: >0.8 good, >0.9 excellent
```

**Key sections in output:**
1. **Engine Performance**: Inclusion rates by engine
2. **SEO Factor Correlations**: Which factors predict inclusion
3. **Predictive Models**: Machine learning performance
4. **Statistical Significance**: Engine differences

#### `analyze_ai_citations.py` - AI Citation Analysis
```bash
# Run analysis
python analyze_ai_citations.py

# Output interpretation:
# - Citation order patterns: Lower numbers = higher priority
# - Content quality differences: Included vs excluded
# - Source authority: Most frequently cited domains
# - Query preferences: Which query types get more citations
```

**Key sections in output:**
1. **Citation Patterns**: Order vs inclusion relationships
2. **Content Quality**: Factors that earn citations
3. **Source Analysis**: Authority and trustworthiness patterns
4. **Query Types**: Citation preferences by category

#### `analyze_combined_insights.py` - Strategic Synthesis
```bash
# Run analysis
python analyze_combined_insights.py

# Output interpretation:
# - Cross-system comparisons: Where systems agree/disagree
# - Universal principles: Factors that work everywhere
# - Strategic recommendations: Dual-track optimization approach
# - Data quality flags: Issues requiring attention
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Selenium/Chrome Issues
```bash
# Error: Chrome driver not found
pip install undetected-chromedriver

# Error: Chrome browser not found
# Install Chrome browser from https://www.google.com/chrome/

# Error: Permission denied
chmod +x .venv/bin/python
```

#### 2. Module Import Errors
```bash
# Error: No module named 'selenium'
pip install -r requirements.txt

# Error: No module named 'undetected_chromedriver'
pip install undetected-chromedriver

# Error: No module named 'distutils'
pip install setuptools
```

#### 3. Data Quality Issues
```bash
# Warning: High inclusion rates (>95%)
# This indicates parsing errors - check CSS selectors
python debug_perplexity.py  # For Perplexity issues

# Warning: Low inclusion rates (<5%)
# This indicates selector failures - update selectors
python debug_google_ai.py   # For Google AI issues
```

#### 4. Visualization Issues
```bash
# Error: matplotlib backend issues
export MPLBACKEND=Agg  # Use non-interactive backend

# Error: Display not available
# All scripts use matplotlib.use('Agg') - should work headless
```

#### 5. File Path Issues
```bash
# Error: File not found
ls -la ai_serp_analysis.csv  # Verify CSV exists
ls -la ../queries/seo_aso_prompts.txt  # Verify queries exist

# Error: Permission denied
chmod 755 *.py  # Make scripts executable
```

### Performance Optimization

#### Memory Usage
```bash
# For large datasets, monitor memory usage
python -c "import psutil; print(f'Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB')"

# If memory issues occur:
# 1. Process data in chunks
# 2. Use pandas.read_csv(chunksize=1000)
# 3. Close unused matplotlib figures
```

#### Speed Optimization
```bash
# Skip already processed queries (edit scrape_geo.py)
# Enable parallel processing for analysis (edit analysis scripts)
# Use SSD storage for faster file I/O
```

## 📈 Interpreting Results

### Traditional SEO Analysis Results

#### Engine Performance
- **Google AI 0.3% inclusion**: AI Overview selectors need updating
- **Bing AI 8.7% inclusion**: Most reliable current data
- **Statistical significance p < 0.05**: Engine differences are real

#### SEO Factor Correlations
- **Page Rank +0.167**: Traditional ranking still matters
- **H2 Count +0.210**: Structured content helps AI inclusion
- **Word Count -0.071**: Longer content less likely to be included

#### Model Performance
- **Accuracy >0.98**: Very good prediction capability
- **Page Rank importance 0.790**: Most important factor
- **Cross-validation consistent**: Results are reliable

### AI Citation Analysis Results

#### Citation Patterns
- **Citation #1-3**: Highest inclusion rates (>90%)
- **Citation #4+**: Declining inclusion probability
- **Order matters**: Earlier citations more likely included

#### Content Quality
- **AI Overview Length**: Most important factor (0.324 importance)
- **Authority sources**: Government, medical, educational sites preferred
- **Content freshness**: Recent content cited more often

#### Query Types
- **"Best-of" queries**: 100% inclusion rate
- **"How-to" queries**: 97.8% inclusion rate
- **"Informational" queries**: 80.0% inclusion rate

### Combined Insights Results

#### Strategic Recommendations
1. **Dual-track optimization**: Different strategies for each system type
2. **Traditional SEO**: Focus on page rank and structured content
3. **AI Citations**: Focus on authority, freshness, and factual accuracy
4. **No universal strategy**: Systems operate differently

#### Data Quality Assessment
- **Green indicators**: Data is reliable for analysis
- **Yellow/Red indicators**: Data quality issues require attention
- **Recommendations**: Prioritize fixing data collection issues

## 🔄 Regular Usage Workflow

### Weekly Analysis Update
```bash
# 1. Update query list if needed
nano ../queries/seo_aso_prompts.txt

# 2. Collect fresh data
python scrape_geo.py

# 3. Process new data
python parse_geo.py

# 4. Generate updated insights
python analyze_traditional_seo.py
python analyze_ai_citations.py
python analyze_combined_insights.py

# 5. Check plots directory for new visualizations
ls -la plots/
```

### Research Workflow
```bash
# 1. Define research questions
# 2. Add relevant queries to query file
# 3. Run data collection
# 4. Analyze results
# 5. Document findings
# 6. Share visualizations and insights
```

### Production Monitoring
```bash
# Set up automated monitoring (future enhancement)
# 1. Schedule daily scraping
# 2. Monitor data quality metrics
# 3. Alert on significant changes
# 4. Generate automated reports
```

## 🎯 Best Practices

### Data Collection
1. **Run scraping during off-peak hours** to reduce bot detection
2. **Monitor data quality metrics** after each collection
3. **Keep backup copies** of successful scrapes
4. **Document any scraping failures** for troubleshooting

### Analysis
1. **Always check data quality first** before drawing conclusions
2. **Validate statistical significance** for all findings
3. **Cross-validate machine learning models** to ensure reliability
4. **Document assumptions and limitations** in analysis

### Visualization
1. **Save all plots** for documentation and sharing
2. **Use consistent color schemes** across visualizations
3. **Include data quality indicators** in dashboards
4. **Provide clear interpretation guides** for non-technical stakeholders

This guide should help you run the complete SEO vs AISO analysis pipeline successfully. For additional questions or issues, refer to the METHODOLOGY.md document or check the individual script documentation.