import os, re, pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs

PERPLEXITY_DIR = "perplexity_search_results_html"
GOOGLE_AI_DIR  = "google_ai_search_results_html"
BING_AI_DIR    = "bing_ai_search_results_html"
OUTPUT_FILE    = "ai_serp_analysis.csv"

results = []

# ----------------------------- helpers ---------------------------------
def norm(u: str) -> str:
    u = str(u).lower()
    u = re.sub(r'^https?://', '', u).lstrip('www.').rstrip('/')
    u = u.split('?', 1)[0].split('#', 1)[0]   # drop ?query and #frag
    return u

def real_href(raw):
    if raw.startswith("https://www.google.com/url?"):
        return parse_qs(urlparse(raw).query).get("q", [""])[0]
    return raw
# ----------------------------------------------------------------------

def extract_seo_elements(path: str, engine: str):
    with open(path, "r", encoding="utf-8") as fh:
        soup = BeautifulSoup(fh, "html.parser")

        title = soup.title.string.strip() if soup.title else ""
        meta   = soup.find("meta", attrs={"name": "description"})
        meta_desc = meta["content"].strip() if meta and meta.get("content") else ""

        h1_tags = [h.get_text(strip=True) for h in soup.find_all("h1")]
        h2_tags = [h.get_text(strip=True) for h in soup.find_all("h2")]
        h3_tags = [h.get_text(strip=True) for h in soup.find_all("h3")]

        canonical = soup.find("link", rel="canonical")
        canonical_url = canonical["href"] if canonical and canonical.get("href") else ""

        word_count = len(re.findall(r"\w+", soup.get_text()))
        anchor_debug = soup.select('ul.zVKf0d a[href]')



        # -------- locate overview TAG (not text yet) -------------------
        if engine == "Google AI":
            overview_tag = soup.select_one("div.LGOjhe, div.ifM9O, div.KpMaL, div.SPZz6b, ul.zVKf0d")
            
        elif engine == "Bing AI":
            overview_tag = soup.select_one("cib-serp, div.b_factrow, div.dg_b, div.b_vlist2col")
        elif engine == "Perplexity":
            overview_tag = soup.select_one("div.prose, div.gap-y-md")
        else:
            overview_tag = None

        overview_text = overview_tag.get_text(" ", strip=True) if overview_tag else ""

        # -------- citation anchors (use real href) ---------------------
        citations = []
        if overview_tag and hasattr(overview_tag, "find_all"):
            for idx, a in enumerate(overview_tag.find_all("a", href=True), 1):
                cleaned = norm(real_href(a["href"]))
                citations.append((idx, cleaned))

        # -------- iterate SERP result containers -----------------------
        for rank, node in enumerate(soup.select("div.tF2Cxc, li.b_algo, div.result"), 1):
            a_tag = node.find("a", href=True)
            link  = a_tag["href"] if a_tag else ""
            link_t= a_tag.get_text(strip=True) if a_tag else ""
            snippet = node.get_text(" ", strip=True)

            link_norm = norm(link)
            included  = any(link_norm == href for _, href in citations)
            order     = next((i for i, href in citations if href == link_norm), None)

            para_idx = None
            if included and overview_tag:
                anchor = next((a for a in overview_tag.find_all("a", href=True)
                               if norm(real_href(a["href"])) == link_norm), None)
                if anchor:
                    parent = anchor.find_parent(["p", "li"])
                    if parent and parent.parent:
                        para_idx = parent.parent.find_all(parent.name).index(parent) + 1


            overview_text = overview_tag.get_text(" ", strip=True) if overview_tag else ""


            results.append({
                "Engine": engine,
                "File": path,
                "Title": title,
                "Meta Description": meta_desc,
                "MetaDesc Length": len(meta_desc),
                "H1 Tags": ", ".join(h1_tags),
                "H2 Tags": ", ".join(h2_tags),
                "H3 Tags": ", ".join(h3_tags),
                "H1 Count": len(h1_tags),
                "H2 Count": len(h2_tags),
                "H3 Count": len(h3_tags),
                "Has H1": int(len(h1_tags) > 0),
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
                "Citation_Paragraph": para_idx
            })

# -------------- walk folders & save -----------------------------------
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
