#!/usr/bin/env python3
import os
from bs4 import BeautifulSoup

# Test file
test_file = "perplexity_search_results_html/AI_tools_for_students_perplexity.html"

if os.path.exists(test_file):
    with open(test_file, "r", encoding="utf-8") as fh:
        soup = BeautifulSoup(fh, "lxml")
        
        print("=== Testing Perplexity HTML Structure ===")
        
        # Test 1: Find all links
        print(f"\n1. Total links found: {len(soup.find_all('a'))}")
        
        # Test 2: Find external links
        external_links = soup.find_all('a', href=True)
        external_links = [a for a in external_links if a['href'].startswith('http') and 'perplexity' not in a['href']]
        print(f"2. External links found: {len(external_links)}")
        for i, link in enumerate(external_links[:5]):
            print(f"   {i+1}. {link.get('href', '')} - {link.get_text(strip=True)[:50]}")
        
        # Test 3: Find divs with specific classes
        print(f"\n3. Divs with 'prose' class: {len(soup.find_all('div', class_='prose'))}")
        print(f"   Divs with 'gap-y-md' class: {len(soup.find_all('div', class_='gap-y-md'))}")
        print(f"   Divs with 'text-base' class: {len(soup.find_all('div', class_='text-base'))}")
        
        # Test 4: Find divs containing 'text' in class name
        text_divs = soup.find_all('div', class_=lambda x: x and 'text' in x)
        print(f"4. Divs with 'text' in class: {len(text_divs)}")
        
        # Test 5: Find divs containing 'content' in class name
        content_divs = soup.find_all('div', class_=lambda x: x and 'content' in x)
        print(f"5. Divs with 'content' in class: {len(content_divs)}")
        
        # Test 6: Find all divs with any class
        all_divs = soup.find_all('div', class_=True)
        print(f"6. Total divs with classes: {len(all_divs)}")
        
        # Test 7: Look for specific patterns
        print(f"\n7. Testing specific selectors:")
        selectors = [
            "div.prose",
            "div.gap-y-md", 
            "div[class*='text']",
            "div[class*='content']",
            "div[class*='answer']",
            "div[class*='response']",
            "a[href^='http']",
            "a[target='_blank']"
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            print(f"   {selector}: {len(elements)} elements")
            
else:
    print(f"Test file {test_file} not found")



