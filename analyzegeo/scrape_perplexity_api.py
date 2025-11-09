#!/usr/bin/env python3
"""
Perplexity API integration for SEO vs AISO analysis.
Replaces web scraping with proper API calls.
"""
import os
import json
import time
import requests
from typing import List, Dict, Any
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
API_KEY = os.getenv('PERPLEXITY_API_KEY')
QUERIES_FILE = '/Users/stephaniekim/development/seoaso/geoseo_analysis/queries/seo_aso_prompts.txt'
OUTPUT_DIR = 'perplexity_api_results'
OUTPUT_CSV = 'perplexity_api_analysis.csv'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_queries() -> List[str]:
    """Load queries from the text file."""
    try:
        with open(QUERIES_FILE, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Query file not found: {QUERIES_FILE}")
        return []

def query_perplexity_api(query: str) -> Dict[str, Any]:
    """
    Query the Perplexity API and return the response.

    Args:
        query: Search query string

    Returns:
        Dictionary containing API response data
    """
    if not API_KEY:
        raise ValueError("PERPLEXITY_API_KEY environment variable not set")

    url = "https://api.perplexity.ai/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-sonar-small-128k-online",  # Use their search model
        "messages": [
            {
                "role": "system",
                "content": "Provide a comprehensive answer with citations to reliable sources."
            },
            {
                "role": "user",
                "content": query
            }
        ],
        "max_tokens": 1000,
        "temperature": 0.2,
        "return_citations": True,
        "return_images": False
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API request failed for query '{query}': {e}")
        return {}

def extract_citations_from_response(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract citation information from Perplexity API response.

    Args:
        response: API response dictionary

    Returns:
        List of citation dictionaries with URL, title, and other metadata
    """
    citations = []

    if not response or 'choices' not in response:
        return citations

    # Extract citations from the response
    if 'citations' in response:
        for i, citation in enumerate(response['citations'], 1):
            citations.append({
                'rank': i,
                'url': citation.get('url', ''),
                'title': citation.get('title', ''),
                'snippet': citation.get('snippet', ''),
                'included': True  # All API citations are included by definition
            })

    # Alternative: extract from message content if citations are embedded
    if not citations and response.get('choices'):
        message = response['choices'][0].get('message', {})
        content = message.get('content', '')

        # Look for citation patterns in content (fallback)
        # This is a simplified approach - the API should return proper citations
        import re
        citation_pattern = r'\[(\d+)\]'
        citation_numbers = re.findall(citation_pattern, content)

        for i, num in enumerate(citation_numbers, 1):
            citations.append({
                'rank': i,
                'url': f'citation_{num}',  # Placeholder - would need proper URL extraction
                'title': f'Citation {num}',
                'snippet': '',
                'included': True
            })

    return citations

def process_all_queries():
    """
    Process all queries through Perplexity API and save results.
    """
    queries = load_queries()

    if not queries:
        print("No queries found to process")
        return

    all_results = []

    for i, query in enumerate(queries, 1):
        print(f"Processing query {i}/{len(queries)}: {query}")

        # Query the API
        response = query_perplexity_api(query)

        if not response:
            print(f"  Skipping due to API error")
            continue

        # Save raw response
        query_filename = query.replace(' ', '_').replace('/', '_')
        with open(f"{OUTPUT_DIR}/{query_filename}_response.json", 'w') as f:
            json.dump(response, f, indent=2)

        # Extract answer content
        answer_content = ""
        if response.get('choices'):
            answer_content = response['choices'][0].get('message', {}).get('content', '')

        # Extract citations
        citations = extract_citations_from_response(response)

        # Create result records
        for citation in citations:
            result = {
                'Engine': 'Perplexity',
                'Query': query,
                'AI_Answer': answer_content,
                'AI_Answer_Length': len(answer_content),
                'Page_Rank': citation['rank'],
                'URL': citation['url'],
                'Title': citation['title'],
                'Snippet': citation['snippet'],
                'Included': citation['included'],
                'Citation_Order': citation['rank']
            }
            all_results.append(result)

        print(f"  Found {len(citations)} citations")

        # Rate limiting - be respectful to the API
        time.sleep(1)

    # Save to CSV
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ Saved {len(all_results)} results to {OUTPUT_CSV}")
        print(f"📊 Processed {len(queries)} queries")
        print(f"📈 Average citations per query: {len(all_results)/len(queries):.1f}")
    else:
        print("❌ No results to save")

if __name__ == "__main__":
    if not API_KEY:
        print("❌ Please set PERPLEXITY_API_KEY environment variable")
        print("   export PERPLEXITY_API_KEY='your-api-key-here'")
        exit(1)

    process_all_queries()