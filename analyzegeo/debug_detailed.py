#!/usr/bin/env python3
import os
import re
from bs4 import BeautifulSoup

def norm(u: str) -> str:
    """Normalize URL for comparison."""
    return re.sub(r'https?://(www\.)?', '', u.lower().split('#')[0].split('?')[0].rstrip('/'))

def real_href(raw: str) -> str:
    """Extract real href from redirect URLs."""
    if 'google.com/url?' in raw:
        try:
            from urllib.parse import parse_qs, urlparse
            parsed = urlparse(raw)
            params = parse_qs(parsed.query)
            return params.get('q', [params.get('url', [raw])[0]])[0]
        except:
            return raw
    return raw

# Test file
test_file = "perplexity_search_results_html/AI_tools_for_students_perplexity.html"

if os.path.exists(test_file):
    with open(test_file, "r", encoding="utf-8") as fh:
        soup = BeautifulSoup(fh, "lxml")
        
        print("=== Detailed Perplexity Debug ===")
        
        # Test the overview tag selector
        overview_selectors = [
            "div.prose",
            "div.gap-y-md", 
            "div[class*='text-base']",
            "div[class*='text-textMain']",
            "div[class*='markdown']",
            "div[class*='content']",
            "div[class*='prose']"
        ]
        
        print("\n1. Testing overview tag selectors:")
        overview_tag = None
        for selector in overview_selectors:
            elements = soup.select(selector)
            print(f"   {selector}: {len(elements)} elements")
            if elements and not overview_tag:
                overview_tag = elements[0]
                print(f"   -> Using first element from {selector}")
        
        if overview_tag:
            print(f"\n2. Overview tag found: {overview_tag.name} with classes: {overview_tag.get('class', [])}")
            overview_text = overview_tag.get_text(" ", strip=True)
            print(f"   Text length: {len(overview_text)}")
            print(f"   First 100 chars: {overview_text[:100]}...")
            
            # Test citation extraction
            citations = []
            for idx, a in enumerate(overview_tag.find_all("a", href=True), 1):
                href = a.get("href", "")
                if href.startswith('http') and 'perplexity' not in href:
                    citations.append((idx, norm(real_href(href))))
            
            print(f"\n3. Citations found in overview: {len(citations)}")
            for idx, (order, href) in enumerate(citations[:5]):
                print(f"   {idx+1}. Order: {order}, URL: {href}")
        else:
            print("\n2. No overview tag found!")
        
        # Test result selectors
        result_selectors = "a[href^='http']:not([href*='perplexity']), a[target='_blank']:not([href*='perplexity'])"
        result_elements = soup.select(result_selectors)
        
        print(f"\n4. Result elements found: {len(result_elements)}")
        for i, element in enumerate(result_elements[:5]):
            href = element.get('href', '')
            text = element.get_text(strip=True)
            print(f"   {i+1}. {href} - {text[:50]}...")
            
        # Test if we would create any rows
        if overview_tag and result_elements:
            print(f"\n5. Would create {len(result_elements)} rows")
        else:
            print(f"\n5. No rows would be created (overview_tag: {overview_tag is not None}, results: {len(result_elements)})")
            
else:
    print(f"Test file {test_file} not found")



