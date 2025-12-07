#!/usr/bin/env python3
"""
Analyze Traditional SEO Factors for AI Overview Inclusion

This script analyzes whether traditional SEO factors predict inclusion
in AI Overviews for Google AI and Bing AI.

Usage:
    python scripts/04_analyze_traditional_seo.py

Prerequisites:
    - data/processed/ai_serp_analysis.csv (from parse_citations.py)
"""

from src.analysis.traditional_seo import (
    load_traditional_seo_data,
    analyze_traditional_ranking_factors,
    analyze_by_engine,
    predictive_modeling
)
from src.analysis.statistical import audit_data_quality
from src.visualization.dashboards import create_traditional_seo_dashboard


def main():
    """Main analysis execution."""
    print("🔍 Traditional SEO Analysis: Google AI + Bing AI")
    print("="*60)

    # Load data
    df = load_traditional_seo_data()

    # Data quality audit
    if 'Included' in df.columns:
        audit_data_quality(df, "All Engines")

        # Per-engine audit
        for engine in df['Engine'].unique():
            engine_df = df[df['Engine'] == engine]
            audit_data_quality(engine_df, engine)

    # Analyze SEO factors
    result = analyze_traditional_ranking_factors(df)
    if result and len(result) == 3:
        correlations, available_features, correlation_results = result
    else:
        correlations = result[0] if result else None
        available_features = result[1] if result and len(result) > 1 else []
        correlation_results = None

    # Engine comparison
    engine_analysis = analyze_by_engine(df)

    # Predictive modeling
    models = predictive_modeling(df, available_features)

    # Create dashboard
    create_traditional_seo_dashboard(df, correlations, engine_analysis, models)

    print(f"\n✅ Traditional SEO Analysis Complete!")
    print(f"📁 Results saved to outputs/figures/traditional_seo_analysis.png")
    print(f"🎯 Focus: Traditional search engines with AI enhancements")


if __name__ == "__main__":
    main()
