#!/usr/bin/env python3
"""
Engine Structure Analysis: Understanding the fundamental differences between
Perplexity, Google AI, and Bing AI for SEO vs AISO research.
"""
import pandas as pd
import numpy as np

def analyze_engine_structures():
    """Analyze the fundamental structural differences between engines."""

    print("🔍 ANALYZING ENGINE STRUCTURE DIFFERENCES")
    print("="*60)

    # Load data
    df = pd.read_csv('ai_serp_analysis.csv')

    print("📊 DATASET OVERVIEW:")
    print(f"Total records: {len(df):,}")
    print(f"Engines: {', '.join(df['Engine'].unique())}")
    print(f"Unique queries: {df['Query_Name'].nunique() if 'Query_Name' in df.columns else 'N/A'}")

    # Analyze each engine's structure
    for engine in df['Engine'].unique():
        engine_data = df[df['Engine'] == engine]
        print(f"\n🔧 {engine.upper()} ANALYSIS:")
        print(f"   Records: {len(engine_data):,}")

        # Check for SERP ranking structure
        print(f"   Page Rank Range: {engine_data['Page Rank'].min()}-{engine_data['Page Rank'].max()}")
        print(f"   Has Traditional SERP Rankings: {'Yes' if engine_data['Page Rank'].max() > 10 else 'Limited to 10'}")

        # Analyze URL diversity
        unique_urls = engine_data['Result URL'].nunique() if 'Result URL' in engine_data.columns else 0
        print(f"   Unique URLs: {unique_urls}")
        print(f"   URL Diversity: {unique_urls / len(engine_data):.2f} (unique URLs per result)")

        # Check AI Overview presence
        has_ai_overview = engine_data['AI Overview'].notna().any() if 'AI Overview' in engine_data.columns else False
        print(f"   Has AI Overview: {has_ai_overview}")

        # Check citation structure
        if 'Citation_Order' in engine_data.columns:
            max_citations = engine_data['Citation_Order'].max()
            print(f"   Max Citations: {max_citations}")

        # Title structure analysis
        if 'Result Title' in engine_data.columns:
            empty_titles = engine_data['Result Title'].isna().sum()
            print(f"   Empty Result Titles: {empty_titles}/{len(engine_data)} ({empty_titles/len(engine_data):.1%})")

def identify_fundamental_differences():
    """Identify the core differences that make engines incomparable."""

    print("\n🚨 FUNDAMENTAL STRUCTURAL DIFFERENCES:")
    print("="*60)

    df = pd.read_csv('ai_serp_analysis.csv')

    # Perplexity Analysis
    perplexity = df[df['Engine'] == 'Perplexity']
    google_ai = df[df['Engine'] == 'Google AI']
    bing_ai = df[df['Engine'] == 'Bing AI']

    print("🔵 PERPLEXITY:")
    print("   Structure: AI-First Citation System")
    print("   What it returns: AI-generated answer with source citations")
    print("   Ranking: Citations ordered by relevance to AI answer (NOT search ranking)")
    print("   SERP: NO traditional search results - only citations within AI response")
    print("   Page content: All results from Perplexity.ai domain (not actual source pages)")

    if len(perplexity) > 0:
        perplexity_urls = perplexity['Result URL'].value_counts().head()
        print(f"   URL Pattern: All results are citations, not ranked web pages")
        print(f"   Citation Count Range: {perplexity['Citation_Order'].min()}-{perplexity['Citation_Order'].max()}")

    print("\n🔵 GOOGLE AI:")
    print("   Structure: Traditional SERP + AI Overview")
    print("   What it returns: Regular search results + AI-generated overview")
    print("   Ranking: Traditional PageRank-based search rankings")
    print("   SERP: YES traditional 10-blue-links + AI Overview on top")
    print("   Page content: Actual web pages from diverse domains")

    if len(google_ai) > 0:
        google_domains = google_ai['Result URL'].str.extract(r'https?://([^/]+)')[0].value_counts().head()
        print(f"   Domain Diversity: {google_ai['Result URL'].str.extract(r'https?://([^/]+)')[0].nunique()} unique domains")

    print("\n🔵 BING AI:")
    print("   Structure: Traditional SERP + AI Chat")
    print("   What it returns: Regular search results + Copilot AI response")
    print("   Ranking: Traditional search rankings")
    print("   SERP: YES traditional search results + AI chat interface")
    print("   Page content: Actual web pages from diverse domains")

    if len(bing_ai) > 0:
        bing_domains = bing_ai['Result URL'].str.extract(r'https?://([^/]+)')[0].value_counts().head()
        print(f"   Domain Diversity: {bing_ai['Result URL'].str.extract(r'https?://([^/]+)')[0].nunique()} unique domains")

def create_comparison_plan():
    """Create a plan for handling the different engine types."""

    print("\n💡 ANALYSIS PLAN RECOMMENDATIONS:")
    print("="*60)

    print("🎯 THE CORE PROBLEM:")
    print("   Perplexity is fundamentally different from Google AI and Bing AI.")
    print("   It's not a search engine - it's an AI answer engine with citations.")
    print("   Comparing 'Page Rank' across engines is meaningless because:")
    print("     • Google AI/Bing AI: Page Rank = traditional search ranking (1-10+)")
    print("     • Perplexity: 'Page Rank' = citation order in AI answer (not search rank)")

    print("\n🔧 METHODOLOGICAL OPTIONS:")

    print("\n   OPTION 1: Exclude Perplexity from Ranking Analysis")
    print("   ✅ Pros:")
    print("     • Maintains scientific validity")
    print("     • Google AI vs Bing AI is a fair comparison (both have traditional SERPs)")
    print("     • Can still analyze Perplexity separately for citation patterns")
    print("   ❌ Cons:")
    print("     • Reduces sample size")
    print("     • Loses insights into AI-first search behavior")

    print("\n   OPTION 2: Create Separate Analysis Tracks")
    print("   ✅ Pros:")
    print("     • Traditional SEO Analysis: Google AI + Bing AI")
    print("     • AI Citation Analysis: Perplexity only")
    print("     • Keeps all data while maintaining validity")
    print("   ❌ Cons:")
    print("     • More complex analysis")
    print("     • Need different metrics for each track")

    print("\n   OPTION 3: Reframe the Research Question")
    print("   ✅ Pros:")
    print("     • Focus on 'content inclusion in AI responses' vs 'search ranking'")
    print("     • All engines become comparable on inclusion metrics")
    print("     • More relevant to AISO (AI Search Optimization)")
    print("   ❌ Cons:")
    print("     • Different from original SEO vs AISO framing")
    print("     • May need different feature analysis")

    print("\n🎯 RECOMMENDED APPROACH:")
    print("   DUAL-TRACK ANALYSIS:")
    print("   📊 Track 1: Traditional SEO Analysis")
    print("     • Engines: Google AI + Bing AI only")
    print("     • Focus: Page ranking, traditional SEO factors")
    print("     • Question: Do traditional SEO factors predict inclusion in AI Overviews?")
    print("   ")
    print("   🤖 Track 2: AI Citation Analysis")
    print("     • Engines: Perplexity (+ Google AI/Bing AI citations if available)")
    print("     • Focus: Citation selection, content quality factors")
    print("     • Question: What content characteristics get cited by AI systems?")

    print("\n📋 IMPLEMENTATION PLAN:")
    print("   1. Keep all existing code and data (as requested)")
    print("   2. Create separate analysis scripts:")
    print("      • analyze_traditional_seo.py (Google AI + Bing AI)")
    print("      • analyze_ai_citations.py (Perplexity focus)")
    print("      • analyze_combined_insights.py (synthesis)")
    print("   3. Update visualizations to reflect the dual-track approach")
    print("   4. Modify research questions to match the methodology")

def main():
    """Main analysis execution."""
    analyze_engine_structures()
    identify_fundamental_differences()
    create_comparison_plan()

    print("\n✅ ANALYSIS COMPLETE")
    print("📋 Next Step: Review recommendations and decide on approach")

if __name__ == "__main__":
    main()