#!/usr/bin/env python3
"""
Comprehensive analysis report for SEO vs AISO study.
"""
import pandas as pd
import numpy as np
from datetime import datetime

def generate_comprehensive_report():
    """Generate a comprehensive analysis report."""

    print("📊 SEO vs AISO: COMPREHENSIVE ANALYSIS REPORT")
    print("=" * 60)
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load data
    df = pd.read_csv('ai_serp_analysis.csv')
    df['Included'] = df['Included'].astype(int)
    df['Query_Name'] = df['File'].str.extract(r'([^/]+)\.html$')[0].str.replace('_', ' ')

    def categorize_query(query):
        query = str(query).lower()
        if query.startswith(('how to', 'how do', 'how does')):
            return 'How-to'
        elif query.startswith(('what is', 'what are', 'what causes')):
            return 'Informational'
        elif ' vs ' in query:
            return 'Comparison'
        elif query.startswith('best '):
            return 'Best-of'
        else:
            return 'Other'

    df['Query_Category'] = df['Query_Name'].apply(categorize_query)

    # SECTION 1: DATASET OVERVIEW
    print("1️⃣ DATASET OVERVIEW")
    print("-" * 30)
    total_results = len(df)
    unique_queries = len(df['Query_Name'].unique())
    engines = df['Engine'].unique()

    print(f"📊 Total Results: {total_results:,}")
    print(f"🔍 Unique Queries: {unique_queries}")
    print(f"🔧 Engines Analyzed: {', '.join(engines)}")
    print(f"📝 Query Categories: {', '.join(df['Query_Category'].unique())}")
    print()

    # SECTION 2: ENGINE PERFORMANCE
    print("2️⃣ ENGINE PERFORMANCE ANALYSIS")
    print("-" * 35)

    engine_stats = df.groupby('Engine').agg({
        'Included': ['count', 'sum', 'mean'],
        'Word Count': 'mean',
        'Page Rank': 'mean'
    }).round(3)

    print("Engine Statistics:")
    for engine in engines:
        engine_df = df[df['Engine'] == engine]
        inclusion_rate = engine_df['Included'].mean()
        total_results = len(engine_df)
        included_count = engine_df['Included'].sum()
        avg_rank = engine_df['Page Rank'].mean()

        print(f"\n🔧 {engine}:")
        print(f"   • Results: {total_results:,}")
        print(f"   • Included: {included_count:,} ({inclusion_rate:.1%})")
        print(f"   • Avg Page Rank: {avg_rank:.1f}")

        # Data quality assessment
        if inclusion_rate > 0.95:
            print(f"   ⚠️ WARNING: Inclusion rate suspiciously high (>95%)")
        elif inclusion_rate < 0.05:
            print(f"   ⚠️ WARNING: Inclusion rate suspiciously low (<5%)")
        else:
            print(f"   ✅ Inclusion rate within normal range")

    print()

    # SECTION 3: QUERY TYPE ANALYSIS
    print("3️⃣ QUERY TYPE PERFORMANCE")
    print("-" * 30)

    query_performance = df.groupby('Query_Category').agg({
        'Included': ['count', 'mean'],
        'Query_Name': 'nunique'
    }).round(3)

    query_performance.columns = ['Total_Results', 'Inclusion_Rate', 'Unique_Queries']
    query_performance = query_performance.sort_values('Inclusion_Rate', ascending=False)

    print("Query Category Performance:")
    for category, row in query_performance.iterrows():
        print(f"\n📝 {category}:")
        print(f"   • Unique Queries: {row['Unique_Queries']}")
        print(f"   • Total Results: {row['Total_Results']}")
        print(f"   • Inclusion Rate: {row['Inclusion_Rate']:.1%}")

    print()

    # SECTION 4: TOP INSIGHTS
    print("4️⃣ KEY INSIGHTS & RECOMMENDATIONS")
    print("-" * 40)

    # Best performers
    best_engine = df.groupby('Engine')['Included'].mean().idxmax()
    best_query_type = query_performance.index[0]
    worst_query_type = query_performance.index[-1]

    print("🏆 TOP PERFORMERS:")
    print(f"   • Best Engine: {best_engine}")
    print(f"   • Best Query Type: {best_query_type}")
    print(f"   • Worst Query Type: {worst_query_type}")

    # Sample high-performing queries
    print(f"\n🎯 HIGH-PERFORMING INDIVIDUAL QUERIES:")
    top_queries = df.groupby('Query_Name')['Included'].agg(['mean', 'count'])
    top_queries = top_queries[top_queries['count'] >= 3]  # At least 3 results
    top_queries = top_queries.sort_values('mean', ascending=False).head(10)

    for i, (query, row) in enumerate(top_queries.iterrows(), 1):
        if row['mean'] > 0:  # Only show queries with some inclusion
            print(f"   {i:2}. {query}: {row['mean']:.1%} ({row['count']} results)")

    print()

    # SECTION 5: DATA QUALITY ISSUES
    print("5️⃣ DATA QUALITY ASSESSMENT")
    print("-" * 35)

    # Check for problematic patterns
    issues_found = []

    # High inclusion rate issue
    perplexity_rate = df[df['Engine'] == 'Perplexity']['Included'].mean() if 'Perplexity' in engines else 0
    if perplexity_rate > 0.95:
        issues_found.append(f"Perplexity inclusion rate too high ({perplexity_rate:.1%}) - likely parsing issue")

    # Low inclusion rate issue
    google_rate = df[df['Engine'] == 'Google AI']['Included'].mean() if 'Google AI' in engines else 0
    if google_rate < 0.05:
        issues_found.append(f"Google AI inclusion rate too low ({google_rate:.1%}) - AI Overviews may not be appearing")

    # Missing data
    missing_cols = []
    important_cols = ['Word Count', 'H1 Count', 'H2 Count', 'MetaDesc Length']
    for col in important_cols:
        if col in df.columns:
            missing_pct = df[col].isna().mean()
            if missing_pct > 0.1:
                missing_cols.append(f"{col}: {missing_pct:.1%} missing")

    if issues_found:
        print("⚠️ ISSUES IDENTIFIED:")
        for issue in issues_found:
            print(f"   • {issue}")
    else:
        print("✅ No major data quality issues detected")

    if missing_cols:
        print(f"\n📋 MISSING DATA:")
        for col in missing_cols:
            print(f"   • {col}")

    print()

    # SECTION 6: RECOMMENDATIONS
    print("6️⃣ RECOMMENDATIONS")
    print("-" * 25)

    recommendations = []

    if perplexity_rate > 0.95:
        recommendations.append("🔧 CRITICAL: Implement Perplexity API to get accurate citation data")

    if google_rate < 0.05:
        recommendations.append("🔧 HIGH: Update Google AI selectors - AI Overviews may have changed")

    if unique_queries < 50:
        recommendations.append(f"📈 MEDIUM: Expand query dataset (currently {unique_queries}, recommend 100+)")

    recommendations.extend([
        "📊 Implement A/B testing across different content optimization strategies",
        "🎯 Focus optimization efforts on 'How-to' and 'Best-of' query types",
        "🔍 Investigate why 'Informational' queries perform poorly",
        "📈 Run longitudinal analysis to track changes over time"
    ])

    for i, rec in enumerate(recommendations, 1):
        print(f"{i:2}. {rec}")

    print()

    # SECTION 7: NEXT STEPS
    print("7️⃣ NEXT STEPS")
    print("-" * 15)

    next_steps = [
        "Run Perplexity API script to get clean citation data",
        "Investigate Google AI low inclusion with updated selectors",
        "Add 50+ more diverse queries to dataset",
        "Implement content optimization experiments",
        "Set up automated monitoring for inclusion rate changes"
    ]

    for i, step in enumerate(next_steps, 1):
        print(f"{i}. {step}")

    print()
    print("="*60)
    print("📈 End of Analysis Report")
    print("="*60)

if __name__ == "__main__":
    generate_comprehensive_report()