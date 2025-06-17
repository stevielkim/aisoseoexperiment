import os
import re
import pandas as pd
from bs4 import BeautifulSoup

# Directories for saved HTML files
PERPLEXITY_DIR = 'perplexity_search_results_html'
GOOGLE_AI_DIR = 'google_ai_search_results_html'
BING_AI_DIR = 'bing_ai_search_results_html'
OUTPUT_FILE = 'ai_serp_analysis.csv'

# Create a DataFrame to store the extracted data
results = []

# Function to extract SEO elements from an HTML file
def extract_seo_elements(filepath, engine):
    with open(filepath, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
        title = soup.title.string.strip() if soup.title else ''
        meta_desc = ''
        meta_tag = soup.find('meta', attrs={'name': 'description'})
        if meta_tag and 'content' in meta_tag.attrs:
            meta_desc = meta_tag['content'].strip()
        h1_tags = [h1.get_text().strip() for h1 in soup.find_all('h1')]
        h2_tags = [h2.get_text().strip() for h2 in soup.find_all('h2')]
        h3_tags = [h3.get_text().strip() for h3 in soup.find_all('h3')]
        word_count = len(re.findall(r'\w+', soup.get_text()))
        canonical_tag = soup.find('link', rel='canonical')
        canonical_url = canonical_tag['href'] if canonical_tag and 'href' in canonical_tag.attrs else ''
        
        # Extract page ranking based on appearance order
        result_containers = soup.select('div.tF2Cxc, li.b_algo, div.result')
        for rank, container in enumerate(result_containers, start=1):
            link_tag = container.find('a')
            link_url = link_tag['href'] if link_tag and 'href' in link_tag.attrs else ''
            link_text = link_tag.get_text().strip() if link_tag else ''
            snippet = container.get_text(separator=' ').strip()
            
            # Extract AI overview (if present)
            ai_overview = ''
            if engine == 'Google AI':
                # Attempt multiple possible containers for Google AI Overview
                ai_overview_element = soup.select_one('div.LGOjhe, div.ifM9O, div.KpMaL, div.SPZz6b, div.vk_c')
                ai_overview = ai_overview_element.get_text(separator=' ').strip() if ai_overview_element else ''
            elif engine == 'Bing AI':
                # Bing AI containers
                ai_overview_element = soup.select_one('div.b_factrow, div.dg_b, div.b_vlist2col')
                ai_overview = ai_overview_element.get_text(separator=' ').strip() if ai_overview_element else ''
            elif engine == 'Perplexity':
                # Perplexity containers
                ai_overview_element = soup.select_one('div.gap-y-md', 'div.prose')
                ai_overview = ai_overview_element.get_text(separator=' ').strip() if ai_overview_element else ''
            
            results.append({
                'Engine': engine,
                'File': filepath,
                'Title': title,
                'Meta Description': meta_desc,
                'H1 Tags': ', '.join(h1_tags),
                'H2 Tags': ', '.join(h2_tags),
                'H3 Tags': ', '.join(h3_tags),
                'Canonical URL': canonical_url,
                'Word Count': word_count,
                'Page Rank': rank,
                'Result Title': link_text,
                'Result URL': link_url,
                'Snippet': snippet,
                'AI Overview': ai_overview
            })

# Extract data from Perplexity HTML files
for filename in os.listdir(PERPLEXITY_DIR):
    if filename.endswith('.html'):
        extract_seo_elements(os.path.join(PERPLEXITY_DIR, filename), 'Perplexity')

# Extract data from Google AI HTML files
for filename in os.listdir(GOOGLE_AI_DIR):
    if filename.endswith('.html'):
        extract_seo_elements(os.path.join(GOOGLE_AI_DIR, filename), 'Google AI')

# Extract data from Bing AI HTML files
for filename in os.listdir(BING_AI_DIR):
    if filename.endswith('.html'):
        extract_seo_elements(os.path.join(BING_AI_DIR, filename), 'Bing AI')

# Save the results to a CSV file
df = pd.DataFrame(results)
df.to_csv(OUTPUT_FILE, index=False)

print(f"AI SERP analysis saved to {OUTPUT_FILE}.")
