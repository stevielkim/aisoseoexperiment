#!/usr/bin/env python3
"""
Debug script to inspect Google AI HTML structure and find working selectors.

This script examines actual Google AI HTML files to:
1. Identify CAPTCHA pages
2. Find AI Overview containers
3. Locate citation links within AI Overview
4. Test various selector candidates
"""
import os
from bs4 import BeautifulSoup
import re

def inspect_google_html(filepath):
    """Inspect Google AI HTML to find AI Overview selectors and citations."""

    filename = os.path.basename(filepath)
    file_size_kb = os.path.getsize(filepath) / 1024

    print(f"\n{'='*80}")
    print(f"File: {filename}")
    print(f"Size: {file_size_kb:.1f} KB")
    print(f"{'='*80}")

    with open(filepath, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    # Check for CAPTCHA
    page_text = soup.get_text().lower()
    if "unusual traffic" in page_text or "captcha" in page_text:
        print("❌ CAPTCHA PAGE DETECTED")
        print("   This file is a CAPTCHA block, not actual search results")
        return

    print(f"✅ Valid search results page ({file_size_kb:.1f} KB)")

    # Look for "AI Overview" text to confirm it's present
    ai_overview_text_found = False
    for text in soup.find_all(string=re.compile(r'AI Overview', re.IGNORECASE)):
        ai_overview_text_found = True
        print(f"\n✓ Found 'AI Overview' text in page")
        break

    if not ai_overview_text_found:
        print("\n⚠️  No 'AI Overview' text found - AI Overview may not be present")

    # Test various AI Overview selectors
    print("\n" + "="*80)
    print("TESTING AI OVERVIEW SELECTORS")
    print("="*80)

    selectors_to_test = [
        # Current selectors from parse_citations.py
        ('div[data-attrid="AnswerV2"]', 'Current: data-attrid="AnswerV2"'),
        ('div.LGOjhe', 'Current: class LGOjhe'),
        ('div.xpdopen', 'Current: class xpdopen'),
        ('div.ifM9O', 'Current: class ifM9O'),
        ('div.kp-whole-page-card', 'Current: class kp-whole-page-card'),
        ('div[data-huuid]', 'Current: data-huuid attribute'),

        # New 2024/2025 selectors to try
        ('div[data-attrid="SGEResponse"]', 'New: SGE Response'),
        ('div[jsname="yQGNIe"]', 'New: jsname yQGNIe'),
        ('c-wiz[jsname="yQGNIe"]', 'New: c-wiz jsname yQGNIe'),
        ('div[data-initq]', 'New: data-initq'),
        ('div.xpdopen div[data-initq]', 'New: xpdopen + data-initq'),

        # Generic AI-related selectors
        ('div[class*="AI"]', 'Generic: class contains AI'),
        ('div[class*="overview"]', 'Generic: class contains overview'),
        ('div[data-attrid*="AI"]', 'Generic: data-attrid contains AI'),
    ]

    found_selectors = []

    for selector, description in selectors_to_test:
        elements = soup.select(selector)
        if elements:
            print(f"\n✅ {description}")
            print(f"   Selector: {selector}")
            print(f"   Found: {len(elements)} element(s)")

            # Show content preview
            first_elem = elements[0]
            text = first_elem.get_text(strip=True)[:200]
            print(f"   Content preview: {text}...")

            # Count links in this container
            links = first_elem.find_all('a', href=True)
            external_links = [a for a in links if a.get('href', '').startswith('http')]
            print(f"   Total links: {len(links)}")
            print(f"   External links: {len(external_links)}")

            if external_links:
                print(f"   First external link: {external_links[0].get('href')[:80]}...")
                found_selectors.append((selector, description, len(external_links)))
        else:
            print(f"❌ {description}: Not found")

    # Search for elements with AI-related attributes
    print("\n" + "="*80)
    print("SEARCHING FOR AI-RELATED ELEMENTS")
    print("="*80)

    ai_keywords = ['sge', 'ai', 'overview', 'generated', 'gemini', 'answer']
    for keyword in ai_keywords:
        # Search in class names
        class_elements = soup.find_all(attrs={'class': lambda x: x and keyword in ' '.join(x).lower()})
        if class_elements:
            print(f"\n✓ Elements with '{keyword}' in class: {len(class_elements)}")
            if class_elements:
                classes = class_elements[0].get('class', [])
                print(f"   Example classes: {' '.join(classes[:3])}")

        # Search in data attributes
        data_elements = soup.find_all(attrs={lambda k: k.startswith('data-') and keyword in k.lower(): True})
        if data_elements:
            print(f"✓ Elements with '{keyword}' in data-attribute: {len(data_elements)}")

    # Look for citation/source indicators
    print("\n" + "="*80)
    print("SEARCHING FOR CITATION INDICATORS")
    print("="*80)

    citation_patterns = [
        ('Sources', 'Text: Sources'),
        ('Learn more', 'Text: Learn more'),
        ('[1]', 'Inline citation [1]'),
        ('[2]', 'Inline citation [2]'),
        ('Show more', 'Text: Show more'),
    ]

    for pattern, description in citation_patterns:
        found = soup.find_all(string=re.compile(re.escape(pattern)))
        if found:
            print(f"✓ {description}: Found {len(found)} instances")
            # Try to find parent container
            if found:
                parent = found[0].find_parent('div')
                if parent:
                    print(f"   Parent div classes: {parent.get('class', [])}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if found_selectors:
        print(f"\n✓ Found {len(found_selectors)} working selector(s):")
        for selector, desc, link_count in found_selectors:
            print(f"   • {desc}: {link_count} external links")
            print(f"     Selector: {selector}")
    else:
        print("\n❌ No working selectors found")
        print("   Manual inspection of HTML needed")

    return found_selectors


def main():
    """Test Google AI HTML files to find working selectors."""

    print("="*80)
    print("GOOGLE AI OVERVIEW SELECTOR DEBUGGING")
    print("="*80)
    print("\nThis script inspects Google AI HTML files to find working selectors")
    print("for extracting AI Overview citations.\n")

    html_dir = "google_ai_search_results_html"

    if not os.path.exists(html_dir):
        print(f"❌ Directory not found: {html_dir}")
        print("   Run this script from the analyzegeo/ directory")
        return

    # Get all HTML files
    html_files = [f for f in os.listdir(html_dir) if f.endswith('.html')]

    if not html_files:
        print(f"❌ No HTML files found in {html_dir}")
        return

    print(f"Found {len(html_files)} HTML files\n")

    # Test on a sample of files - mix of different sizes
    test_files = []

    # Get files by size
    file_sizes = []
    for filename in html_files:
        filepath = os.path.join(html_dir, filename)
        size = os.path.getsize(filepath)
        file_sizes.append((filename, size, filepath))

    file_sizes.sort(key=lambda x: x[1])

    # Sample: smallest (likely CAPTCHA), medium, and largest files
    if len(file_sizes) >= 3:
        test_files = [
            file_sizes[0][2],  # Smallest
            file_sizes[len(file_sizes)//2][2],  # Middle
            file_sizes[-1][2],  # Largest
        ]
    else:
        test_files = [f[2] for f in file_sizes]

    # Add a few more random samples
    import random
    random.seed(42)
    sample = random.sample(html_files, min(3, len(html_files)))
    for filename in sample:
        filepath = os.path.join(html_dir, filename)
        if filepath not in test_files:
            test_files.append(filepath)

    print(f"Testing {len(test_files)} sample files:")
    for f in test_files:
        print(f"  - {os.path.basename(f)}")

    # Inspect each file
    all_working_selectors = []
    for filepath in test_files:
        working = inspect_google_html(filepath)
        if working:
            all_working_selectors.extend(working)

    # Final recommendations
    print("\n" + "="*80)
    print("FINAL RECOMMENDATIONS")
    print("="*80)

    if all_working_selectors:
        # Count which selectors worked most frequently
        from collections import Counter
        selector_counts = Counter([s[0] for s in all_working_selectors])

        print("\n✅ Recommended selectors to add to parse_citations.py:")
        print("   (Listed in order of reliability)\n")

        for selector, count in selector_counts.most_common():
            print(f"   '{selector}',  # Worked in {count}/{len(test_files)} files")
    else:
        print("\n❌ No working selectors found in any files")
        print("\nNext steps:")
        print("   1. Manually open 1-2 HTML files in a browser")
        print("   2. Inspect the AI Overview section with browser DevTools")
        print("   3. Note the class names and data attributes")
        print("   4. Update parse_citations.py selectors manually")

    print("\n" + "="*80)
    print("DEBUGGING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
