# Perplexity API Setup Guide

## Why Use the API Instead of Web Scraping?

- **Reliable data structure**: JSON responses vs parsing dynamic JavaScript
- **Better citation quality**: Proper source attribution and metadata
- **No bot detection**: APIs don't require bypassing scraping protection
- **Faster execution**: Direct API calls vs waiting for page rendering
- **More stable**: API endpoints change less than HTML selectors

## Setup Steps

### 1. Get Perplexity API Access

1. Visit [Perplexity AI API](https://docs.perplexity.ai/)
2. Sign up for an API account
3. Generate an API key from your dashboard
4. Note: There may be rate limits and costs associated

### 2. Set Up Environment Variables

#### Option A: Using .env file (Recommended)
```bash
# Edit the .env file that was created for you
nano .env

# Replace 'your_perplexity_api_key_here' with your actual API key:
# PERPLEXITY_API_KEY=pplx-your-actual-api-key-here
```

#### Option B: Using shell environment variables
```bash
# Export the API key (replace with your actual key)
export PERPLEXITY_API_KEY='pplx-your-api-key-here'

# Or add to your shell profile (~/.bashrc, ~/.zshrc)
echo 'export PERPLEXITY_API_KEY="pplx-your-api-key-here"' >> ~/.zshrc
source ~/.zshrc
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
# This includes: requests, python-dotenv, and other required packages
```

### 4. Test the API

```bash
# Test with a single query
python -c "
import os
from scrape_perplexity_api import query_perplexity_api
response = query_perplexity_api('how to improve sleep')
print(f'Status: {\"Success\" if response else \"Failed\"}')
if response:
    print(f'Response keys: {list(response.keys())}')
"
```

### 5. Run Full Collection

```bash
# Process all queries (this will take time due to rate limiting)
python scrape_perplexity_api.py
```

## Expected Output

The script will create:
- `perplexity_api_results/` directory with JSON responses
- `perplexity_api_analysis.csv` with structured data
- Console output showing progress

## Integration with Existing Pipeline

The new CSV format matches the existing analysis pipeline:
- Same column structure as parsed HTML data
- Drop-in replacement for web scraping results
- Works with existing `analyze_geo.py` analysis script

## API Response Structure

```json
{
  "choices": [{
    "message": {
      "content": "AI-generated answer with citations [1][2][3]",
      "role": "assistant"
    }
  }],
  "citations": [
    {
      "url": "https://example.com/source1",
      "title": "Source 1 Title",
      "snippet": "Relevant excerpt..."
    }
  ]
}
```

## Rate Limiting

- Built-in 1-second delay between requests
- Adjust `time.sleep(1)` in script if needed
- Monitor API quotas and usage

## Troubleshooting

### API Key Issues
```bash
# Check if key is set
echo $PERPLEXITY_API_KEY

# Should show: pplx-...
```

### Request Failures
- Check API quota/billing
- Verify key permissions
- Check internet connection
- Review API documentation for changes