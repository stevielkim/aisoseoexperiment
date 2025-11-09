#!/usr/bin/env python3
"""
Debug script to analyze Perplexity HTML parsing issues.
"""
import os
from bs4 import BeautifulSoup
import pandas as pd

def test_perplexity_parsing():
    """Test Perplexity HTML parsing and show what we're actually extracting."""

    # Get first Perplexity file
    perp_dir = "perplexity_search_results_html"
    files = [f for f in os.listdir(perp_dir) if f.endswith(".html")]

    if not files:
        print("No Perplexity files found!")
        return

    test_file = os.path.join(perp_dir, files[0])
    print(f"Testing file: {test_file}")

    with open(test_file, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    print("\n" + "="*50)
    print("CURRENT PERPLEXITY PARSING ANALYSIS")
    print("="*50)

    # Test overview selectors
    overview_selectors = ["div.prose", "div.gap-y-md", "div[data-testid]", ".answer", ".response"]

    print("\n1. Testing AI Overview selectors:")
    for selector in overview_selectors:
        elements = soup.select(selector)
        print(f"   {selector}: {len(elements)} elements found")
        if elements:
            text = elements[0].get_text(" ", strip=True)[:100] + "..." if len(elements[0].get_text()) > 100 else elements[0].get_text(" ", strip=True)
            print(f"      First element text: {text}")

    # Test result selectors
    result_selectors = [
        "a[href^='http']:not([href*='perplexity'])",
        "a[target='_blank']:not([href*='perplexity'])",
        ".citation",
        "[data-testid*='citation']",
        ".source"
    ]

    print("\n2. Testing citation/result selectors:")
    for selector in result_selectors:
        elements = soup.select(selector)
        print(f"   {selector}: {len(elements)} elements found")
        if elements:
            for i, elem in enumerate(elements[:3]):  # Show first 3
                href = elem.get('href', 'No href')
                text = elem.get_text(strip=True)[:50] + "..." if len(elem.get_text()) > 50 else elem.get_text(strip=True)
                print(f"      [{i+1}] href: {href}")
                print(f"          text: {text}")

    # Look for any divs with specific classes/ids that might contain content
    print("\n3. Searching for content containers:")
    for tag in soup.find_all(['div', 'section', 'article'], limit=20):
        class_names = ' '.join(tag.get('class', []))
        id_name = tag.get('id', '')
        if class_names or id_name:
            text = tag.get_text(strip=True)
            if len(text) > 50:  # Only show divs with substantial content
                print(f"   Tag: {tag.name}, Class: {class_names[:50]}, ID: {id_name[:20]}")
                print(f"        Content: {text[:80]}...")

    # Check for JSON data in script tags
    print("\n4. Checking for JSON data in script tags:")
    scripts = soup.find_all('script')
    json_scripts = [s for s in scripts if s.string and ('answer' in s.string.lower() or 'sources' in s.string.lower() or 'citations' in s.string.lower())]
    print(f"   Found {len(json_scripts)} scripts with relevant keywords")

    if json_scripts:
        print("   Sample script content (first 200 chars):")
        print(f"   {json_scripts[0].string[:200]}...")

    print("\n5. Summary:")
    print(f"   File size: {os.path.getsize(test_file)} bytes")
    print(f"   Total text length: {len(soup.get_text())} characters")
    print(f"   Total links: {len(soup.find_all('a'))}")
    print(f"   External links: {len(soup.select('a[href^=\"http\"]'))}")

if __name__ == "__main__":
    test_perplexity_parsing()