#!/usr/bin/env python3
"""
Parse Citations: Extract cited URLs from AI search results.

This script extracts citations from AI Overview responses.
Focus: Clean extraction of cited URLs for content analysis.

Note: Google AI and Bing AI data has CAPTCHA issues.
Perplexity data is currently the most reliable.
"""
import os
import re
import json
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
from collections import defaultdict

# Directories
PERPLEXITY_DIR = "perplexity_search_results_html"
GOOGLE_AI_DIR = "google_ai_search_results_html"
BING_AI_DIR = "bing_ai_search_results_html"
OUTPUT_FILE = "citations_clean.csv"


def normalize_url(url: str) -> str:
    """Normalize URL for comparison."""
    if not url:
        return ""
    url = str(url).strip().lower()
    url = re.sub(r'^https?://', '', url)
    url = url.split('?')[0].split('#')[0]
    url = url.lstrip('www.').rstrip('/')
    return url


def extract_query_from_filename(filename: str) -> str:
    """Extract query name from filename."""
    # Remove engine suffix and extension
    name = filename.replace('_perplexity.html', '')
    name = name.replace('_google_ai.html', '')
    name = name.replace('_bing_ai.html', '')
    # Convert underscores to spaces
    return name.replace('_', ' ')


def is_captcha_page(soup: BeautifulSoup) -> bool:
    """Check if page is a CAPTCHA challenge."""
    text = soup.get_text().lower()
    captcha_indicators = [
        'recaptcha',
        'captcha',
        'verify you are human',
        'unusual traffic',
        'automated queries'
    ]
    return any(indicator in text for indicator in captcha_indicators)


def parse_perplexity_citations(filepath: str) -> list:
    """
    Extract citations from Perplexity HTML.
    
    Perplexity structure:
    - Citations are inline with class="citation inline"
    - Each citation has an <a> tag with href to source
    """
    citations = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
    
    query = extract_query_from_filename(os.path.basename(filepath))
    
    # Find all citation elements
    citation_elements = soup.select('span.citation.inline a[href]')
    
    # Also try alternative selectors
    if not citation_elements:
        citation_elements = soup.select('.citation a[href]')
    
    if not citation_elements:
        # Fallback: find all external links in the prose area
        prose = soup.select_one('div.prose')
        if prose:
            citation_elements = prose.find_all('a', href=True)
    
    # Track unique URLs per query
    seen_urls = set()
    citation_order = 0
    
    for elem in citation_elements:
        url = elem.get('href', '')
        
        # Skip internal links, javascript, etc.
        if not url.startswith(('http://', 'https://')):
            continue
        
        # Skip Perplexity internal links
        if 'perplexity.ai' in url:
            continue
        
        url_normalized = normalize_url(url)
        
        # Skip duplicates within same query
        if url_normalized in seen_urls:
            continue
        
        seen_urls.add(url_normalized)
        citation_order += 1
        
        citations.append({
            'query': query,
            'engine': 'Perplexity',
            'citation_order': citation_order,
            'url': url,
            'url_normalized': url_normalized,
            'source_file': os.path.basename(filepath),
            'is_valid': True,
        })
    
    return citations


def parse_google_ai_citations(filepath: str) -> list:
    """
    Extract citations from Google AI Overview HTML.
    
    Note: Many files are CAPTCHA pages and will be flagged.
    """
    citations = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
    
    query = extract_query_from_filename(os.path.basename(filepath))
    
    # Check for CAPTCHA
    if is_captcha_page(soup):
        citations.append({
            'query': query,
            'engine': 'Google AI',
            'citation_order': 0,
            'url': '',
            'url_normalized': '',
            'source_file': os.path.basename(filepath),
            'is_valid': False,
            'error': 'CAPTCHA page',
        })
        return citations
    
    # Try various AI Overview selectors
    overview_selectors = [
        'div[data-initq]',  # 2024/2025 AI Overview selector - WORKING
        'div[data-attrid="AnswerV2"]',
        'div.LGOjhe',
        'div.xpdopen',
        'div.ifM9O',
        'div.kp-whole-page-card',
        'div[data-huuid]',
    ]
    
    overview_tag = None
    for selector in overview_selectors:
        overview_tag = soup.select_one(selector)
        if overview_tag:
            break
    
    if not overview_tag:
        # No AI Overview found
        citations.append({
            'query': query,
            'engine': 'Google AI',
            'citation_order': 0,
            'url': '',
            'url_normalized': '',
            'source_file': os.path.basename(filepath),
            'is_valid': False,
            'error': 'No AI Overview found',
        })
        return citations
    
    # Extract citation links from AI Overview
    seen_urls = set()
    citation_order = 0
    
    for a in overview_tag.find_all('a', href=True):
        raw_url = a.get('href', '')
        
        # Handle Google redirect URLs
        if raw_url.startswith('https://www.google.com/url?'):
            try:
                url = parse_qs(urlparse(raw_url).query).get('q', [''])[0]
            except:
                url = raw_url
        else:
            url = raw_url
        
        # Skip internal links
        if not url.startswith(('http://', 'https://')):
            continue
        if 'google.com' in url:
            continue
        
        url_normalized = normalize_url(url)
        
        if url_normalized in seen_urls:
            continue
        
        seen_urls.add(url_normalized)
        citation_order += 1
        
        citations.append({
            'query': query,
            'engine': 'Google AI',
            'citation_order': citation_order,
            'url': url,
            'url_normalized': url_normalized,
            'source_file': os.path.basename(filepath),
            'is_valid': True,
        })
    
    if not citations:
        citations.append({
            'query': query,
            'engine': 'Google AI',
            'citation_order': 0,
            'url': '',
            'url_normalized': '',
            'source_file': os.path.basename(filepath),
            'is_valid': False,
            'error': 'No citations found in AI Overview',
        })
    
    return citations


def parse_bing_ai_citations(filepath: str) -> list:
    """
    Extract citations from Bing AI HTML.
    
    Note: Many files are CAPTCHA pages and will be flagged.
    """
    citations = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
    
    query = extract_query_from_filename(os.path.basename(filepath))
    
    # Check for CAPTCHA
    if is_captcha_page(soup):
        citations.append({
            'query': query,
            'engine': 'Bing AI',
            'citation_order': 0,
            'url': '',
            'url_normalized': '',
            'source_file': os.path.basename(filepath),
            'is_valid': False,
            'error': 'CAPTCHA page',
        })
        return citations
    
    # Try Bing AI selectors
    overview_selectors = [
        'div.qna-mf',
        'div.b_ans',
        'div#b_results div.b_ans',
    ]
    
    overview_tag = None
    for selector in overview_selectors:
        overview_tag = soup.select_one(selector)
        if overview_tag:
            break
    
    if not overview_tag:
        citations.append({
            'query': query,
            'engine': 'Bing AI',
            'citation_order': 0,
            'url': '',
            'url_normalized': '',
            'source_file': os.path.basename(filepath),
            'is_valid': False,
            'error': 'No AI Overview found',
        })
        return citations
    
    # Extract citation links
    seen_urls = set()
    citation_order = 0
    
    for a in overview_tag.find_all('a', href=True):
        url = a.get('href', '')
        
        if not url.startswith(('http://', 'https://')):
            continue
        if 'bing.com' in url or 'microsoft.com' in url:
            continue
        
        url_normalized = normalize_url(url)
        
        if url_normalized in seen_urls:
            continue
        
        seen_urls.add(url_normalized)
        citation_order += 1
        
        citations.append({
            'query': query,
            'engine': 'Bing AI',
            'citation_order': citation_order,
            'url': url,
            'url_normalized': url_normalized,
            'source_file': os.path.basename(filepath),
            'is_valid': True,
        })
    
    if not citations:
        citations.append({
            'query': query,
            'engine': 'Bing AI',
            'citation_order': 0,
            'url': '',
            'url_normalized': '',
            'source_file': os.path.basename(filepath),
            'is_valid': False,
            'error': 'No citations found in AI Overview',
        })
    
    return citations


def main():
    """Parse all HTML files and extract citations."""
    print("=" * 60)
    print("📚 PARSE CITATIONS")
    print("   Extracting cited URLs from AI search results")
    print("=" * 60)
    
    all_citations = []
    stats = defaultdict(lambda: {'total': 0, 'valid': 0, 'captcha': 0, 'no_citations': 0})
    
    # Parse Perplexity files
    print("\n🔍 Parsing Perplexity files...")
    if os.path.exists(PERPLEXITY_DIR):
        for filename in sorted(os.listdir(PERPLEXITY_DIR)):
            if filename.endswith('.html'):
                filepath = os.path.join(PERPLEXITY_DIR, filename)
                citations = parse_perplexity_citations(filepath)
                all_citations.extend(citations)
                stats['Perplexity']['total'] += 1
                if citations and citations[0].get('is_valid', False):
                    stats['Perplexity']['valid'] += 1
        print(f"   ✅ Processed {stats['Perplexity']['total']} files")
    
    # Parse Google AI files
    print("\n🔍 Parsing Google AI files...")
    if os.path.exists(GOOGLE_AI_DIR):
        for filename in sorted(os.listdir(GOOGLE_AI_DIR)):
            if filename.endswith('.html'):
                filepath = os.path.join(GOOGLE_AI_DIR, filename)
                citations = parse_google_ai_citations(filepath)
                all_citations.extend(citations)
                stats['Google AI']['total'] += 1
                
                if citations:
                    if citations[0].get('is_valid'):
                        stats['Google AI']['valid'] += 1
                    elif citations[0].get('error') == 'CAPTCHA page':
                        stats['Google AI']['captcha'] += 1
                    else:
                        stats['Google AI']['no_citations'] += 1
        print(f"   ✅ Processed {stats['Google AI']['total']} files")
        print(f"   ⚠️  CAPTCHA pages: {stats['Google AI']['captcha']}")
    
    # Parse Bing AI files  
    print("\n🔍 Parsing Bing AI files...")
    if os.path.exists(BING_AI_DIR):
        for filename in sorted(os.listdir(BING_AI_DIR)):
            if filename.endswith('.html'):
                filepath = os.path.join(BING_AI_DIR, filename)
                citations = parse_bing_ai_citations(filepath)
                all_citations.extend(citations)
                stats['Bing AI']['total'] += 1
                
                if citations:
                    if citations[0].get('is_valid'):
                        stats['Bing AI']['valid'] += 1
                    elif citations[0].get('error') == 'CAPTCHA page':
                        stats['Bing AI']['captcha'] += 1
                    else:
                        stats['Bing AI']['no_citations'] += 1
        print(f"   ✅ Processed {stats['Bing AI']['total']} files")
        print(f"   ⚠️  CAPTCHA pages: {stats['Bing AI']['captcha']}")
    
    # Create DataFrame
    df = pd.DataFrame(all_citations)
    
    # Filter to valid citations only
    valid_df = df[df['is_valid'] == True].copy()
    
    # Save all data
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Saved all citations to {OUTPUT_FILE}")
    
    # Save valid citations only
    valid_df.to_csv('citations_valid.csv', index=False)
    print(f"✅ Saved valid citations to citations_valid.csv")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    for engine in ['Perplexity', 'Google AI', 'Bing AI']:
        s = stats[engine]
        print(f"\n{engine}:")
        print(f"   • Total files: {s['total']}")
        print(f"   • Valid files: {s['valid']}")
        if s['captcha'] > 0:
            print(f"   • CAPTCHA blocked: {s['captcha']} ⚠️")
        if s['no_citations'] > 0:
            print(f"   • No citations found: {s['no_citations']}")
    
    print(f"\n📈 Total valid citations: {len(valid_df)}")
    print(f"   • Unique URLs: {valid_df['url_normalized'].nunique()}")
    print(f"   • Unique queries: {valid_df['query'].nunique()}")
    
    if len(valid_df) > 0:
        print(f"\n🏆 Top cited domains:")
        valid_df['domain'] = valid_df['url_normalized'].apply(lambda x: x.split('/')[0] if x else '')
        top_domains = valid_df['domain'].value_counts().head(10)
        for domain, count in top_domains.items():
            print(f"   • {domain}: {count}")
    
    print(f"\n✅ Done! Next step: Run fetch_source_features.py")


if __name__ == "__main__":
    main()

