#!/usr/bin/env python3
"""
Debug script to inspect Bing HTML structure and verify Copilot presence.

This script examines actual Bing HTML files to:
1. Determine if Copilot content loaded
2. Find Copilot-specific elements
3. Locate citation links within Copilot
4. Test various selector candidates
"""
import os
from bs4 import BeautifulSoup
import re

def inspect_bing_html(filepath):
    """Inspect Bing HTML to determine if Copilot loaded and find citation selectors."""

    filename = os.path.basename(filepath)
    file_size_kb = os.path.getsize(filepath) / 1024

    print(f"\n{'='*80}")
    print(f"File: {filename}")
    print(f"Size: {file_size_kb:.1f} KB")
    print(f"{'='*80}")

    with open(filepath, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    # Check for Copilot-specific elements
    print("\n" + "="*80)
    print("CHECKING FOR COPILOT CONTENT")
    print("="*80)

    copilot_found = False

    # Test Copilot-specific selectors
    copilot_selectors = [
        ('cib-serp', 'Primary: cib-serp custom element'),
        ('cib-conversation', 'Copilot: conversation container'),
        ('cib-message-group', 'Copilot: message group'),
        ('div[class*="cib"]', 'Generic: class contains cib'),
        ('[class^="cib-"]', 'Generic: class starts with cib-'),
    ]

    copilot_elements = []

    for selector, description in copilot_selectors:
        elements = soup.select(selector)
        if elements:
            copilot_found = True
            print(f"✅ {description}")
            print(f"   Selector: {selector}")
            print(f"   Found: {len(elements)} element(s)")

            # Show content preview
            text = elements[0].get_text(strip=True)[:200]
            print(f"   Content preview: {text}...")

            # Count links
            links = elements[0].find_all('a', href=True)
            external_links = [a for a in links if a.get('href', '').startswith('http')]
            print(f"   Links found: {len(external_links)} external")

            copilot_elements.append((selector, description, elements))
        else:
            print(f"❌ {description}: Not found")

    # Search for "Copilot" text
    print("\n" + "="*80)
    print("SEARCHING FOR 'COPILOT' TEXT")
    print("="*80)

    copilot_text = soup.find_all(string=re.compile(r'Copilot', re.IGNORECASE))
    if copilot_text:
        print(f"✓ Found 'Copilot' text: {len(copilot_text)} instances")
        for i, text in enumerate(copilot_text[:3]):
            parent = text.find_parent()
            if parent:
                print(f"   {i+1}. Context: {text.strip()[:60]}...")
                print(f"      Parent tag: <{parent.name}>, classes: {parent.get('class', [])}")
    else:
        print("❌ No 'Copilot' text found in page")

    # Determine if this is regular SERP or Copilot
    print("\n" + "="*80)
    print("CONTENT TYPE ANALYSIS")
    print("="*80)

    # Check for regular SERP indicators
    regular_serp = soup.select('#b_results li.b_algo')
    if regular_serp:
        print(f"✓ Regular SERP results found: {len(regular_serp)} results")
    else:
        print("⚠️  No regular SERP results found")

    # Verdict
    if copilot_found:
        print(f"\n✅ VERDICT: Copilot content IS present in HTML")
        print(f"   File contains Copilot-specific elements")
        print(f"   → This is a PARSER problem (selectors need updating)")
    else:
        print(f"\n❌ VERDICT: Copilot content NOT present in HTML")
        if regular_serp:
            print(f"   File contains only regular SERP results ({len(regular_serp)} items)")
        else:
            print(f"   File may be incomplete or unusual structure")
        print(f"   → This is a SCRAPER problem (Copilot tab didn't load)")

    # If Copilot found, test citation extraction
    if copilot_found:
        print("\n" + "="*80)
        print("TESTING CITATION EXTRACTION FROM COPILOT")
        print("="*80)

        for selector, description, elements in copilot_elements[:1]:  # Test first working selector
            elem = elements[0]

            # Find all links
            all_links = elem.find_all('a', href=True)
            external_links = []

            for a in all_links:
                href = a.get('href', '')
                # Filter external links (skip bing.com, microsoft.com)
                if href.startswith('http') and 'bing.com' not in href and 'microsoft.com' not in href:
                    external_links.append(href)

            print(f"\nFrom selector: {selector}")
            print(f"Total links: {len(all_links)}")
            print(f"External links (potential citations): {len(external_links)}")

            if external_links:
                print(f"\nSample citations found:")
                for i, link in enumerate(external_links[:5], 1):
                    print(f"   {i}. {link[:80]}...")
            else:
                print(f"\n⚠️  No external citation links found in Copilot content")

    # Test legacy SERP selectors (what parser currently uses)
    print("\n" + "="*80)
    print("TESTING LEGACY SERP SELECTORS")
    print("="*80)

    legacy_selectors = [
        ('div.qna-mf', 'Legacy: qna-mf'),
        ('div.b_ans', 'Legacy: b_ans'),
        ('div#b_results div.b_ans', 'Legacy: b_results b_ans'),
    ]

    for selector, description in legacy_selectors:
        elements = soup.select(selector)
        if elements:
            print(f"✓ {description}: Found {len(elements)} elements")
            links = elements[0].find_all('a', href=True)
            print(f"   Links: {len(links)}")
        else:
            print(f"❌ {description}: Not found")

    return copilot_found


def main():
    """Test Bing HTML files to determine if Copilot loaded."""

    print("="*80)
    print("BING COPILOT SELECTOR DEBUGGING")
    print("="*80)
    print("\nThis script inspects Bing HTML files to determine if Copilot content")
    print("loaded and find working selectors for citation extraction.\n")

    html_dir = "bing_ai_search_results_html"

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

    # Sample files to test
    import random
    random.seed(42)
    sample_files = random.sample(html_files, min(5, len(html_files)))

    print(f"Testing {len(sample_files)} sample files:")
    for f in sample_files:
        print(f"  - {f}")

    # Track results
    copilot_found_count = 0
    copilot_not_found_count = 0

    for filename in sample_files:
        filepath = os.path.join(html_dir, filename)
        copilot_found = inspect_bing_html(filepath)

        if copilot_found:
            copilot_found_count += 1
        else:
            copilot_not_found_count += 1

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    print(f"\nTested {len(sample_files)} files:")
    print(f"  ✅ Copilot found: {copilot_found_count}/{len(sample_files)}")
    print(f"  ❌ Copilot not found: {copilot_not_found_count}/{len(sample_files)}")

    if copilot_found_count > 0:
        print("\n✅ RECOMMENDATION: Parser fixes needed")
        print("\nSome files contain Copilot content. Update parse_citations.py with:")
        print("   1. Add cib-serp, cib-conversation selectors")
        print("   2. Extract external links from Copilot elements")
        print("   3. Filter out bing.com and microsoft.com links")

        if copilot_not_found_count > 0:
            print(f"\n⚠️  {copilot_not_found_count} files missing Copilot")
            print("   Consider re-scraping those queries with improved scraper")
    else:
        print("\n❌ RECOMMENDATION: Scraper fixes needed")
        print("\nNO files contain Copilot content. The scraper needs fixing:")
        print("   1. Fix Copilot tab activation in scrape_geo.py")
        print("   2. Verify cib-serp element loads before saving HTML")
        print("   3. Re-scrape all 73 Bing queries")
        print("\nNext steps:")
        print("   1. Manually visit a Bing search in a browser")
        print("   2. Inspect the Copilot tab with DevTools")
        print("   3. Note the tab element's attributes")
        print("   4. Update scrape_geo.py with correct tab selector")

    print("\n" + "="*80)
    print("DEBUGGING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
