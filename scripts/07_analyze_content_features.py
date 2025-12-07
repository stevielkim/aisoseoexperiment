#!/usr/bin/env python3
"""
Analyze Content Features for AI Citations

This script analyzes what content features predict citation likelihood
in AI search engines (Perplexity, Google AI, Bing AI).

Usage:
    python scripts/07_analyze_content_features.py

Prerequisites:
    - data/processed/ai_serp_analysis.csv (from parse_citations.py)
    - data/processed/source_features.csv (from fetch_source_features.py)
"""

from src.analysis.content_features import (
    load_data,
    analyze_citation_patterns,
    analyze_citation_order,
    analyze_domain_patterns,
    analyze_source_content_features,
    analyze_citation_order_vs_features,
    build_feature_importance_model
)
from src.analysis.statistical import audit_data_quality
from src.visualization.dashboards import create_content_feature_dashboard


def main():
    """Main analysis execution."""
    print("=" * 60)
    print("🎯 AI CONTENT CITATION ANALYSIS")
    print("=" * 60)
    print("\nGoal: What features in content make it more or less likely")
    print("      to be included in AI Overviews?\n")

    # Load data
    citations_df, source_df = load_data()

    if citations_df is None:
        return

    # Data quality audit
    if 'Included' in citations_df.columns:
        audit_data_quality(citations_df, "All Engines")

        # Per-engine audit
        for engine in citations_df['Engine'].unique():
            engine_df = citations_df[citations_df['Engine'] == engine]
            audit_data_quality(engine_df, engine)

    # Analyze citation patterns
    engine_stats, valid_df = analyze_citation_patterns(citations_df)

    # Analyze citation order
    order_counts = analyze_citation_order(valid_df)

    # Analyze domain patterns
    domain_counts, type_counts = analyze_domain_patterns(valid_df)

    # Analyze source content features (if available)
    merged_df = None
    feature_stats = None
    available_features = []
    correlations = None
    importance_result = None

    if source_df is not None:
        result = analyze_source_content_features(citations_df, source_df)
        if result:
            merged_df, feature_stats, available_features = result

            # Citation order vs features
            result = analyze_citation_order_vs_features(merged_df, available_features)
            if result and len(result) == 2:
                correlations, correlation_results = result
            else:
                correlations = result if result else None

            # Feature importance model
            importance_result = build_feature_importance_model(merged_df, available_features)

    # Extract importance_df if model was built
    importance_df = None
    if importance_result:
        importance_df, _ = importance_result

    # Create dashboard
    create_content_feature_dashboard(
        citations_df,
        domain_counts,
        type_counts,
        feature_stats,
        correlations,
        importance_df
    )

    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\n📁 Results saved to: outputs/figures/content_feature_analysis.png")
    print("\n💡 Next Steps:")
    if source_df is None:
        print("   1. Run: python scripts/03_extract_features.py")
        print("      (This will fetch actual content features from source URLs)")
        print("   2. Re-run: python scripts/07_analyze_content_features.py")
        print("      (To see full content feature analysis)")
    else:
        print("   1. Review the dashboard for content optimization insights")
        print("   2. Consider expanding the query dataset for more robust analysis")


if __name__ == "__main__":
    main()
