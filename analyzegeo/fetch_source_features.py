#!/usr/bin/env python3
"""
Fetch Source Features: Extract content features from actual source URLs.

This script fetches the actual source pages cited in AI Overviews and extracts
content features that may predict citation likelihood.
"""
import os
import re
import json
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
OUTPUT_FILE = "source_features.csv"
REQUEST_TIMEOUT = 15
REQUEST_DELAY = 1  # Seconds between requests to be polite

# Headers to mimic a real browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}


def extract_domain_info(url):
    """Extract domain-level features from URL."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Remove www prefix
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Domain type classification
        domain_type = 'commercial'
        if domain.endswith('.edu'):
            domain_type = 'educational'
        elif domain.endswith('.gov'):
            domain_type = 'government'
        elif domain.endswith('.org'):
            domain_type = 'organization'
        elif domain.endswith('.io'):
            domain_type = 'tech'
        
        # Check for known authoritative domains
        authoritative_domains = [
            'wikipedia.org', 'mayoclinic.org', 'healthline.com', 'webmd.com',
            'nih.gov', 'cdc.gov', 'harvard.edu', 'stanford.edu', 'mit.edu',
            'nytimes.com', 'bbc.com', 'reuters.com', 'forbes.com'
        ]
        is_authoritative = any(auth in domain for auth in authoritative_domains)
        
        return {
            'domain': domain,
            'domain_type': domain_type,
            'is_authoritative_domain': int(is_authoritative),
            'url_depth': len([p for p in parsed.path.split('/') if p]),
        }
    except Exception as e:
        return {
            'domain': '',
            'domain_type': 'unknown',
            'is_authoritative_domain': 0,
            'url_depth': 0,
        }


def extract_structure_features(soup):
    """Extract structural content features from HTML."""
    # Heading counts
    h1_tags = soup.find_all('h1')
    h2_tags = soup.find_all('h2')
    h3_tags = soup.find_all('h3')
    
    h1_texts = [h.get_text(strip=True) for h in h1_tags]
    h2_texts = [h.get_text(strip=True) for h in h2_tags]
    h3_texts = [h.get_text(strip=True) for h in h3_tags]
    
    # List counts
    ordered_lists = soup.find_all('ol')
    unordered_lists = soup.find_all('ul')
    list_items = soup.find_all('li')
    
    # Table counts
    tables = soup.find_all('table')
    
    # Paragraph analysis
    paragraphs = soup.find_all('p')
    para_texts = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]
    para_lengths = [len(p) for p in para_texts]
    
    avg_para_length = sum(para_lengths) / len(para_lengths) if para_lengths else 0
    short_paras = sum(1 for l in para_lengths if l < 100)
    long_paras = sum(1 for l in para_lengths if l > 300)
    
    return {
        'h1_count': len(h1_tags),
        'h2_count': len(h2_tags),
        'h3_count': len(h3_tags),
        'total_headings': len(h1_tags) + len(h2_tags) + len(h3_tags),
        'ordered_list_count': len(ordered_lists),
        'unordered_list_count': len(unordered_lists),
        'total_list_count': len(ordered_lists) + len(unordered_lists),
        'list_item_count': len(list_items),
        'table_count': len(tables),
        'paragraph_count': len(para_texts),
        'avg_paragraph_length': round(avg_para_length, 1),
        'short_paragraphs': short_paras,
        'long_paragraphs': long_paras,
        'h1_text': h1_texts[0] if h1_texts else '',
    }


def extract_content_depth_features(soup):
    """Extract features related to content depth and comprehensiveness."""
    # Get main content text (try to exclude nav, footer, etc.)
    # Remove script and style elements
    for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
        element.decompose()
    
    text = soup.get_text(separator=' ', strip=True)
    words = re.findall(r'\w+', text)
    
    # Word count
    word_count = len(words)
    
    # Unique words (vocabulary richness)
    unique_words = len(set(w.lower() for w in words))
    vocabulary_richness = unique_words / word_count if word_count > 0 else 0
    
    # Sentence estimation (rough)
    sentences = re.split(r'[.!?]+', text)
    sentence_count = len([s for s in sentences if len(s.strip()) > 10])
    
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    
    # Question detection (indicates Q&A format)
    questions = len(re.findall(r'\?', text))
    
    # Number/statistic density (indicates data-rich content)
    numbers = len(re.findall(r'\b\d+(?:\.\d+)?%?\b', text))
    number_density = numbers / word_count * 100 if word_count > 0 else 0
    
    return {
        'word_count': word_count,
        'unique_word_count': unique_words,
        'vocabulary_richness': round(vocabulary_richness, 3),
        'sentence_count': sentence_count,
        'avg_sentence_length': round(avg_sentence_length, 1),
        'question_count': questions,
        'number_count': numbers,
        'number_density': round(number_density, 2),
    }


def extract_quality_signals(soup, response_headers):
    """Extract quality signals from content and HTTP headers."""
    # Author detection
    author = None
    author_selectors = [
        'meta[name="author"]',
        '[rel="author"]',
        '.author',
        '.byline',
        '[itemprop="author"]',
    ]
    for selector in author_selectors:
        element = soup.select_one(selector)
        if element:
            author = element.get('content') or element.get_text(strip=True)
            if author:
                break
    
    # Date detection
    publish_date = None
    date_selectors = [
        'meta[property="article:published_time"]',
        'meta[name="date"]',
        'meta[name="publish-date"]',
        'time[datetime]',
        '[itemprop="datePublished"]',
    ]
    for selector in date_selectors:
        element = soup.select_one(selector)
        if element:
            publish_date = element.get('content') or element.get('datetime') or element.get_text(strip=True)
            if publish_date:
                break
    
    # Modified date
    modified_date = None
    modified_selectors = [
        'meta[property="article:modified_time"]',
        'meta[name="last-modified"]',
        '[itemprop="dateModified"]',
    ]
    for selector in modified_selectors:
        element = soup.select_one(selector)
        if element:
            modified_date = element.get('content') or element.get_text(strip=True)
            if modified_date:
                break
    
    # Last-Modified header
    last_modified_header = response_headers.get('Last-Modified', '')
    
    return {
        'has_author': int(bool(author)),
        'author_name': author or '',
        'has_publish_date': int(bool(publish_date)),
        'publish_date': publish_date or '',
        'has_modified_date': int(bool(modified_date or last_modified_header)),
        'modified_date': modified_date or last_modified_header or '',
    }


def extract_formatting_features(soup):
    """Extract formatting and media features."""
    # Images
    images = soup.find_all('img')
    images_with_alt = [img for img in images if img.get('alt', '').strip()]
    
    # Links
    internal_links = []
    external_links = []
    all_links = soup.find_all('a', href=True)
    
    for link in all_links:
        href = link.get('href', '')
        if href.startswith(('http://', 'https://')):
            external_links.append(href)
        elif href.startswith('/') or not href.startswith(('#', 'javascript:', 'mailto:')):
            internal_links.append(href)
    
    # Code blocks
    code_blocks = soup.find_all(['code', 'pre'])
    
    # Blockquotes
    blockquotes = soup.find_all('blockquote')
    
    # Videos
    videos = soup.find_all(['video', 'iframe'])
    
    # Bold/emphasis
    bold_elements = soup.find_all(['strong', 'b'])
    
    return {
        'image_count': len(images),
        'images_with_alt': len(images_with_alt),
        'image_alt_ratio': round(len(images_with_alt) / len(images), 2) if images else 0,
        'internal_link_count': len(internal_links),
        'external_link_count': len(external_links),
        'total_link_count': len(all_links),
        'code_block_count': len(code_blocks),
        'blockquote_count': len(blockquotes),
        'video_count': len(videos),
        'bold_element_count': len(bold_elements),
    }


def extract_schema_markup(soup):
    """Extract schema.org structured data."""
    schema_types = {
        'has_faq_schema': 0,
        'has_howto_schema': 0,
        'has_article_schema': 0,
        'has_review_schema': 0,
        'has_product_schema': 0,
        'has_recipe_schema': 0,
        'has_any_schema': 0,
    }
    
    # Check JSON-LD
    for script in soup.find_all('script', type='application/ld+json'):
        try:
            data = json.loads(script.string or '{}')
            items = data if isinstance(data, list) else [data]
            
            for item in items:
                schema_type = item.get('@type', '')
                if isinstance(schema_type, list):
                    types = [t.lower() for t in schema_type]
                else:
                    types = [str(schema_type).lower()]
                
                if 'faqpage' in types:
                    schema_types['has_faq_schema'] = 1
                if 'howto' in types:
                    schema_types['has_howto_schema'] = 1
                if any(t in types for t in ['article', 'newsarticle', 'blogposting']):
                    schema_types['has_article_schema'] = 1
                if 'review' in types:
                    schema_types['has_review_schema'] = 1
                if 'product' in types:
                    schema_types['has_product_schema'] = 1
                if 'recipe' in types:
                    schema_types['has_recipe_schema'] = 1
                    
        except (json.JSONDecodeError, TypeError):
            continue
    
    # Check if any schema exists
    schema_types['has_any_schema'] = int(any(v == 1 for k, v in schema_types.items() if k != 'has_any_schema'))
    
    return schema_types


def detect_content_type(soup, url):
    """Detect the type/format of content."""
    text = soup.get_text(separator=' ', strip=True).lower()
    h1 = soup.find('h1')
    title = h1.get_text(strip=True).lower() if h1 else ''
    url_lower = url.lower()
    
    content_type = 'general'
    
    # How-to detection
    if any(pattern in title or pattern in url_lower for pattern in ['how to', 'how-to', 'guide', 'tutorial', 'step by step']):
        content_type = 'how_to'
    # Listicle detection
    elif re.search(r'\b\d+\s+(best|top|ways|tips|things|reasons)', title):
        content_type = 'listicle'
    # Comparison detection
    elif ' vs ' in title or 'comparison' in title or 'versus' in title:
        content_type = 'comparison'
    # Q&A detection
    elif title.endswith('?') or 'what is' in title or 'what are' in title:
        content_type = 'qa'
    # Review detection
    elif 'review' in title or 'review' in url_lower:
        content_type = 'review'
    # Definition detection
    elif title.startswith('what is') or 'definition' in title:
        content_type = 'definition'
    
    return {
        'content_type': content_type,
        'is_how_to': int(content_type == 'how_to'),
        'is_listicle': int(content_type == 'listicle'),
        'is_comparison': int(content_type == 'comparison'),
        'is_qa': int(content_type == 'qa'),
    }


def fetch_page_features(url):
    """Fetch a URL and extract all content features."""
    features = {
        'url': url,
        'fetch_success': 0,
        'fetch_error': '',
    }
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Mark success
        features['fetch_success'] = 1
        features['status_code'] = response.status_code
        features['final_url'] = response.url
        
        # Extract all feature categories
        features.update(extract_domain_info(url))
        features.update(extract_structure_features(soup))
        features.update(extract_content_depth_features(soup))
        features.update(extract_quality_signals(soup, response.headers))
        features.update(extract_formatting_features(soup))
        features.update(extract_schema_markup(soup))
        features.update(detect_content_type(soup, url))
        
        # Meta information
        title_tag = soup.find('title')
        features['page_title'] = title_tag.get_text(strip=True) if title_tag else ''
        
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        features['meta_description'] = meta_desc.get('content', '') if meta_desc else ''
        features['meta_desc_length'] = len(features['meta_description'])
        
    except requests.exceptions.Timeout:
        features['fetch_error'] = 'timeout'
    except requests.exceptions.HTTPError as e:
        features['fetch_error'] = f'http_{e.response.status_code}'
    except requests.exceptions.RequestException as e:
        features['fetch_error'] = str(type(e).__name__)
    except Exception as e:
        features['fetch_error'] = str(e)[:100]
    
    return features


def load_citation_data():
    """Load the citation data from the AI SERP analysis."""
    csv_path = 'ai_serp_analysis.csv'
    
    if not os.path.exists(csv_path):
        print(f"❌ Citation data not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} records from {csv_path}")
    
    return df


def get_unique_source_urls(df):
    """Extract unique source URLs from citation data."""
    # Get URLs from Result URL column
    if 'Result URL' not in df.columns:
        print("❌ 'Result URL' column not found in data")
        return []
    
    urls = df['Result URL'].dropna().unique()
    
    # Filter out empty and invalid URLs
    valid_urls = [
        url for url in urls 
        if url and 
        isinstance(url, str) and 
        url.startswith(('http://', 'https://')) and
        len(url) > 10
    ]
    
    print(f"📊 Found {len(valid_urls)} unique source URLs to fetch")
    return valid_urls


def main():
    """Main execution: fetch features for all source URLs."""
    print("🔍 Source Feature Extraction")
    print("=" * 60)
    print("Goal: Extract content features from actual source pages")
    print("=" * 60)
    
    # Load citation data
    df = load_citation_data()
    if df is None:
        return
    
    # Get unique URLs
    urls = get_unique_source_urls(df)
    if not urls:
        print("❌ No valid URLs found")
        return
    
    # Check for existing data to resume
    existing_urls = set()
    if os.path.exists(OUTPUT_FILE):
        existing_df = pd.read_csv(OUTPUT_FILE)
        existing_urls = set(existing_df['url'].values)
        print(f"📁 Found {len(existing_urls)} already fetched URLs")
    
    # Filter URLs to fetch
    urls_to_fetch = [u for u in urls if u not in existing_urls]
    print(f"🌐 URLs to fetch: {len(urls_to_fetch)}")
    
    if not urls_to_fetch:
        print("✅ All URLs already fetched!")
        return
    
    # Fetch features for each URL
    results = []
    total = len(urls_to_fetch)
    
    for i, url in enumerate(urls_to_fetch, 1):
        print(f"[{i}/{total}] Fetching: {url[:60]}...")
        
        features = fetch_page_features(url)
        results.append(features)
        
        if features['fetch_success']:
            print(f"    ✅ {features['word_count']} words, {features['h2_count']} H2s, type: {features['content_type']}")
        else:
            print(f"    ❌ Error: {features['fetch_error']}")
        
        # Rate limiting
        if i < total:
            time.sleep(REQUEST_DELAY)
    
    # Combine with existing data
    results_df = pd.DataFrame(results)
    
    if existing_urls:
        existing_df = pd.read_csv(OUTPUT_FILE)
        results_df = pd.concat([existing_df, results_df], ignore_index=True)
    
    # Save results
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Saved {len(results_df)} source features to {OUTPUT_FILE}")
    
    # Summary statistics
    success_count = results_df['fetch_success'].sum()
    print(f"\n📊 Summary:")
    print(f"   • Total sources: {len(results_df)}")
    print(f"   • Successfully fetched: {success_count} ({success_count/len(results_df)*100:.1f}%)")
    print(f"   • Average word count: {results_df[results_df['fetch_success']==1]['word_count'].mean():.0f}")


if __name__ == "__main__":
    main()
