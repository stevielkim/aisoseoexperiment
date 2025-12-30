#!/usr/bin/env python3
import os, re, json, pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs

# Use timestamped directories from latest scraping run
PERPLEXITY_DIR = "data/raw/html/perplexity_search_results_html_20251223_121620"
GOOGLE_AI_DIR  = "data/raw/html/google_ai_search_results_html_20251223_121620"
BING_AI_DIR    = "data/raw/html/bing_ai_search_results_html"  # Skip - no new data
OUTPUT_FILE    = "data/processed/ai_serp_analysis_20251223.csv"

results = []

# ----------------------------- helpers ---------------------------------
Q_WORDS = re.compile(r'^\s*(who|what|where|when|why|how)\b', re.I)

def norm(u: str) -> str:
    u = str(u).strip()
    if not u:
        return ""
    u = u.lower()
    u = re.sub(r'^https?://', '', u)
    u = u.split('?', 1)[0].split('#', 1)[0]
    u = u.lstrip('www.').rstrip('/')
    return u

def real_href(raw: str) -> str:
    raw = raw or ""
    if raw.startswith("https://www.google.com/url?"):
        try:
            return parse_qs(urlparse(raw).query).get("q", [""])[0]
        except Exception:
            return raw
    return raw

def text_len(s):
    return len((s or "").strip())

def count_question_headings(h_list):
    return sum(1 for h in h_list if Q_WORDS.match(h))

def parse_jsonld_schema(soup: BeautifulSoup):
    """Return flags for FAQ, HowTo, Article if present in any JSON-LD block."""
    flags = {"Has FAQ Schema": 0, "Has HowTo Schema": 0, "Has Article Schema": 0}
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "{}")
        except Exception:
            continue
        # normalize into iterable
        items = data if isinstance(data, list) else [data]
        for it in items:
            t = it.get("@type")
            if isinstance(t, list):
                types = [str(x).lower() for x in t]
            else:
                types = [str(t).lower()] if t else []
            if any("faqpage" == x for x in types):
                flags["Has FAQ Schema"] = 1
            if any("howto" == x for x in types):
                flags["Has HowTo Schema"] = 1
            if any(x in ("article","newsarticle","blogposting") for x in types):
                flags["Has Article Schema"] = 1
    return flags

def structure_metrics(soup: BeautifulSoup):
    """Compute structural features from saved HTML."""
    # Headings text
    h1_txt = [h.get_text(strip=True) for h in soup.find_all("h1")]
    h2_txt = [h.get_text(strip=True) for h in soup.find_all("h2")]
    h3_txt = [h.get_text(strip=True) for h in soup.find_all("h3")]

    # Lists / tables
    ols = soup.find_all("ol")
    uls = soup.find_all("ul")
    lis = soup.find_all("li")
    tables = soup.find_all("table")

    # Paragraphs
    paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    para_lengths = [len(p) for p in paras if p]
    avg_para_len = sum(para_lengths)/len(para_lengths) if para_lengths else 0
    short_paras = sum(1 for L in para_lengths if L < 150)

    # Images
    imgs = soup.find_all("img")
    img_count = len(imgs)
    img_alt50 = sum(1 for im in imgs if text_len(im.get("alt")) >= 50)

    # TOC anchors: in-page links like <a href="#section">
    toc_anchors = soup.select('a[href^="#"]')
    has_toc = int(len(toc_anchors) > 3)  # heuristic: more than 3 suggests a TOC

    # Question-style headings
    q_headings = count_question_headings(h1_txt + h2_txt + h3_txt)

    return {
        "H1 Tags": ", ".join(h1_txt),
        "H2 Tags": ", ".join(h2_txt),
        "H3 Tags": ", ".join(h3_txt),
        "H1 Count": len(h1_txt),
        "H2 Count": len(h2_txt),
        "H3 Count": len(h3_txt),
        "Has H1": int(len(h1_txt) > 0),
        "List Count": len(ols) + len(uls),
        "OL Count": len(ols),
        "UL Count": len(uls),
        "List Item Count": len(lis),
        "Table Count": len(tables),
        "Avg Paragraph Length": round(avg_para_len, 1),
        "Short Paragraphs (<150)": short_paras,
        "Image Count": img_count,
        "Images Alt>=50": img_alt50,
        "Question Heading Count": q_headings,
        "Has TOC Anchors": has_toc,
    }

# ----------------------------------------------------------------------
def extract_seo_elements(path: str, engine: str):
    with open(path, "r", encoding="utf-8") as fh:
        soup = BeautifulSoup(fh, "html.parser")

        # Basics
        title = soup.title.string.strip() if soup.title else ""
        meta_tag = soup.find("meta", attrs={"name": "description"})
        meta_desc = meta_tag["content"].strip() if meta_tag and meta_tag.get("content") else ""
        canonical = soup.find("link", rel="canonical")
        canonical_url = canonical["href"] if canonical and canonical.get("href") else ""
        word_count = len(re.findall(r"\w+", soup.get_text()))

        # Structural metrics
        struct = structure_metrics(soup)
        schema_flags = parse_jsonld_schema(soup)

        # Overview container (TAG, not text)
        # REVERTED TO AUGUST 12 LOGIC: Use same container for overview text AND citations
        if engine == "Google AI":
            overview_tag = soup.select_one("div.LGOjhe, div.ifM9O, div.KpMaL, div.SPZz6b, div.vk_c, ul.zVKf0d")
        elif engine == "Bing AI":
            # For Bing, look for iframe content embedded by scraper
            iframe_content = soup.find(string=lambda text: text and "IFRAME_CONTENT_START" in text)
            if iframe_content:
                # Extract and parse the iframe HTML
                iframe_html_match = re.search(r'<!-- IFRAME_CONTENT_START -->(.*?)<!-- IFRAME_CONTENT_END -->',
                                             str(soup), re.DOTALL)
                if iframe_html_match:
                    iframe_soup = BeautifulSoup(iframe_html_match.group(1), 'html.parser')
                    # Look for citations in the iframe (Bing SERP result structure)
                    overview_tag = iframe_soup.select_one("body")  # Get all iframe content
                else:
                    overview_tag = soup.select_one("#b_copilot_search")
            else:
                # Fallback to old method
                overview_tag = soup.select_one("#b_copilot_search .b_cs_canvas, #b_copilot_search .ca_container")
                if not overview_tag:
                    overview_tag = soup.select_one("#b_copilot_search")
        elif engine == "Perplexity":
            overview_tag = soup.select_one("div.prose, div.gap-y-md")
        else:
            overview_tag = None

        overview_text = overview_tag.get_text(" ", strip=True) if overview_tag else ""

        # Citation anchors (use href, strip redirect, normalize)
        # Extract citations from SAME container as overview text
        citations = []
        if overview_tag:
            for idx, a in enumerate(overview_tag.find_all("a", href=True), 1):
                citations.append((idx, norm(real_href(a["href"]))))

        # Iterate SERP result containers (blue links) - Fixed selectors
        if engine == "Google AI":
            result_selectors = "div.tF2Cxc, li.b_algo, div.result"
        elif engine == "Bing AI":
            # Bing uses li.b_algo for search results, gs_sm_cit for AI citations
            result_selectors = "li.b_algo, div.gs_sm_cit, div.gs_sup_cit, div.b_attribution"
        elif engine == "Perplexity":
            result_selectors = ".citation"  # Use the citation elements we found
        else:
            result_selectors = "div.tF2Cxc, li.b_algo, div.result"

        for rank, node in enumerate(soup.select(result_selectors), 1):
            if engine == "Perplexity":
                # For Perplexity citations, find the href attribute or nested link
                link = node.get("href", "")
                if not link:
                    # Look for nested a tag
                    nested_a = node.find("a", href=True)
                    link = nested_a["href"] if nested_a else ""
                a_tag = node if node.get("href") else nested_a
            else:
                a_tag = node.find("a", href=True)
                link = a_tag["href"] if a_tag else ""

            link_t = a_tag.get_text(strip=True) if a_tag else ""
            snippet = node.get_text(" ", strip=True)

            link_norm = norm(link)
            included = any(link_norm == href for _, href in citations)
            order = next((i for i, href in citations if href == link_norm), None)

            # paragraph/list index for included links
            para_idx = None
            if included and overview_tag:
                anchor = next((a for a in overview_tag.find_all("a", href=True)
                               if norm(real_href(a["href"])) == link_norm), None)
                if anchor:
                    parent = anchor.find_parent(["p", "li"])
                    if parent and parent.parent:
                        para_idx = parent.parent.find_all(parent.name).index(parent) + 1

            row = {
                "Engine": engine,
                "File": path,
                "Title": title,
                "Meta Description": meta_desc,
                "MetaDesc Length": len(meta_desc),
                "Canonical URL": canonical_url,
                "Word Count": word_count,
                "Page Rank": rank,
                "Result Title": link_t,
                "Result URL": link,
                "Snippet": snippet,
                "Snippet Length": len(snippet),
                "AI Overview": overview_text,
                "AI Overview Length": len(overview_text),
                "Included": included,
                "Citation_Order": order,
                "Citation_Paragraph": para_idx,
            }

            # merge structural & schema flags
            row.update(struct)
            row.update(schema_flags)
            results.append(row)

# -------------- walk folders & save -----------------------------------
def main():
    for fn in os.listdir(PERPLEXITY_DIR):
        if fn.endswith(".html"):
            extract_seo_elements(os.path.join(PERPLEXITY_DIR, fn), "Perplexity")

    for fn in os.listdir(GOOGLE_AI_DIR):
        if fn.endswith(".html"):
            extract_seo_elements(os.path.join(GOOGLE_AI_DIR, fn), "Google AI")

    for fn in os.listdir(BING_AI_DIR):
        if fn.endswith(".html"):
            extract_seo_elements(os.path.join(BING_AI_DIR, fn), "Bing AI")

    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"✓ Saved enriched AI SERP data → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
