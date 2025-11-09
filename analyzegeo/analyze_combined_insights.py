#!/usr/bin/env python3
"""
Combined Insights Analysis: Synthesis of Traditional SEO + AI Citation Analysis
Focus: What can we learn from both approaches together?
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("Set2")

def load_and_segment_data():
    """Load data and segment by engine type."""
    print("🔄 Loading and Segmenting Data by Engine Type")
    print("="*60)

    df = pd.read_csv('ai_serp_analysis.csv')
    df['Included'] = df['Included'].astype(int)
    df['Query_Name'] = df['File'].str.extract(r'([^/]+)\.html$')[0].str.replace('_', ' ')

    # Categorize queries
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

    # Segment data
    traditional_seo = df[df['Engine'].isin(['Google AI', 'Bing AI'])].copy()
    ai_citations = df[df['Engine'] == 'Perplexity'].copy()

    print(f"📊 Dataset Segmentation:")
    print(f"   • Traditional SEO (Google AI + Bing AI): {len(traditional_seo):,} records")
    print(f"   • AI Citations (Perplexity): {len(ai_citations):,} records")
    print(f"   • Unique queries: {df['Query_Name'].nunique()}")

    return df, traditional_seo, ai_citations

def cross_engine_query_analysis(traditional_seo, ai_citations):
    """Compare how the same queries perform across different engine types."""
    print("\n🔄 CROSS-ENGINE QUERY COMPARISON")
    print("="*50)

    # Find queries present in both datasets
    traditional_queries = set(traditional_seo['Query_Name'].unique())
    ai_queries = set(ai_citations['Query_Name'].unique())
    common_queries = traditional_queries.intersection(ai_queries)

    print(f"📊 Query Coverage:")
    print(f"   • Traditional SEO queries: {len(traditional_queries)}")
    print(f"   • AI Citation queries: {len(ai_queries)}")
    print(f"   • Common queries: {len(common_queries)}")

    if len(common_queries) > 0:
        # Analyze performance for common queries
        comparison_results = []

        for query in common_queries:
            trad_data = traditional_seo[traditional_seo['Query_Name'] == query]
            ai_data = ai_citations[ai_citations['Query_Name'] == query]

            trad_inclusion = trad_data['Included'].mean() if len(trad_data) > 0 else 0
            ai_inclusion = ai_data['Included'].mean() if len(ai_data) > 0 else 0

            comparison_results.append({
                'query': query,
                'traditional_inclusion': trad_inclusion,
                'ai_inclusion': ai_inclusion,
                'difference': ai_inclusion - trad_inclusion,
                'traditional_count': len(trad_data),
                'ai_count': len(ai_data)
            })

        comparison_df = pd.DataFrame(comparison_results)

        print(f"\n🎯 Performance Comparison for Common Queries:")
        print(f"   • Avg Traditional SEO inclusion: {comparison_df['traditional_inclusion'].mean():.1%}")
        print(f"   • Avg AI Citation inclusion: {comparison_df['ai_inclusion'].mean():.1%}")

        # Top queries where AI citations perform much better
        top_ai_advantage = comparison_df.nlargest(5, 'difference')
        print(f"\n🤖 Queries where AI Citations Excel:")
        for _, row in top_ai_advantage.iterrows():
            print(f"   • {row['query']}: AI {row['ai_inclusion']:.1%} vs Traditional {row['traditional_inclusion']:.1%}")

        # Top queries where traditional SEO performs better
        top_traditional_advantage = comparison_df.nsmallest(5, 'difference')
        print(f"\n🔍 Queries where Traditional SEO Excels:")
        for _, row in top_traditional_advantage.iterrows():
            print(f"   • {row['query']}: Traditional {row['traditional_inclusion']:.1%} vs AI {row['ai_inclusion']:.1%}")

        return comparison_df
    else:
        print("❌ No common queries found for comparison")
        return pd.DataFrame()

def analyze_content_optimization_insights(traditional_seo, ai_citations):
    """Derive insights for content optimization across both approaches."""
    print("\n📝 CONTENT OPTIMIZATION INSIGHTS")
    print("="*50)

    # Define content features present in both datasets
    common_features = ['Word Count', 'H1 Count', 'H2 Count', 'H3 Count', 'MetaDesc Length']
    available_features = [f for f in common_features if f in traditional_seo.columns and f in ai_citations.columns]

    print(f"📋 Analyzing common content features: {', '.join(available_features)}")

    insights = {}

    for feature in available_features:
        # Traditional SEO analysis
        trad_included = traditional_seo[traditional_seo['Included'] == 1][feature].dropna()
        trad_excluded = traditional_seo[traditional_seo['Included'] == 0][feature].dropna()

        # AI Citation analysis
        ai_included = ai_citations[ai_citations['Included'] == 1][feature].dropna()
        ai_excluded = ai_citations[ai_citations['Included'] == 0][feature].dropna()

        insights[feature] = {
            'traditional_included_mean': trad_included.mean() if len(trad_included) > 0 else 0,
            'traditional_excluded_mean': trad_excluded.mean() if len(trad_excluded) > 0 else 0,
            'ai_included_mean': ai_included.mean() if len(ai_included) > 0 else 0,
            'ai_excluded_mean': ai_excluded.mean() if len(ai_excluded) > 0 else 0
        }

        # Calculate differences
        trad_diff = insights[feature]['traditional_included_mean'] - insights[feature]['traditional_excluded_mean']
        ai_diff = insights[feature]['ai_included_mean'] - insights[feature]['ai_excluded_mean']

        print(f"\n📊 {feature}:")
        print(f"   Traditional SEO:")
        print(f"     • Included avg: {insights[feature]['traditional_included_mean']:.1f}")
        print(f"     • Excluded avg: {insights[feature]['traditional_excluded_mean']:.1f}")
        print(f"     • Difference: {trad_diff:+.1f}")
        print(f"   AI Citations:")
        print(f"     • Included avg: {insights[feature]['ai_included_mean']:.1f}")
        print(f"     • Excluded avg: {insights[feature]['ai_excluded_mean']:.1f}")
        print(f"     • Difference: {ai_diff:+.1f}")

        # Determine optimization direction
        if abs(trad_diff) > 10 or abs(ai_diff) > 10:  # Meaningful difference threshold
            direction = "increase" if (trad_diff > 0 and ai_diff > 0) else "optimize carefully"
            print(f"     → Recommendation: {direction}")

    return insights

def identify_universal_principles(traditional_seo, ai_citations):
    """Identify principles that work across both traditional SEO and AI citations."""
    print("\n🌐 UNIVERSAL OPTIMIZATION PRINCIPLES")
    print("="*50)

    universal_principles = []

    # Query category analysis across both systems
    trad_query_perf = traditional_seo.groupby('Query_Category')['Included'].mean()
    ai_query_perf = ai_citations.groupby('Query_Category')['Included'].mean()

    print("🎯 Query Type Performance Comparison:")
    for category in set(trad_query_perf.index).intersection(set(ai_query_perf.index)):
        trad_rate = trad_query_perf[category]
        ai_rate = ai_query_perf[category]
        print(f"   • {category}:")
        print(f"     - Traditional SEO: {trad_rate:.1%}")
        print(f"     - AI Citations: {ai_rate:.1%}")

        if trad_rate > 0.1 and ai_rate > 0.1:  # Both perform reasonably well
            universal_principles.append(f"'{category}' queries perform well in both systems")

    # Content length analysis
    if 'Word Count' in traditional_seo.columns and 'Word Count' in ai_citations.columns:
        trad_optimal_length = traditional_seo[traditional_seo['Included'] == 1]['Word Count'].median()
        ai_optimal_length = ai_citations[ai_citations['Included'] == 1]['Word Count'].median()

        print(f"\n📏 Optimal Content Length:")
        print(f"   • Traditional SEO: {trad_optimal_length:.0f} words")
        print(f"   • AI Citations: {ai_optimal_length:.0f} words")

        if abs(trad_optimal_length - ai_optimal_length) / max(trad_optimal_length, ai_optimal_length) < 0.3:
            universal_principles.append(f"Optimal content length similar across systems (~{(trad_optimal_length + ai_optimal_length)/2:.0f} words)")

    print(f"\n✅ UNIVERSAL PRINCIPLES IDENTIFIED:")
    for i, principle in enumerate(universal_principles, 1):
        print(f"   {i}. {principle}")

    return universal_principles

def create_combined_insights_dashboard(df, traditional_seo, ai_citations, comparison_df, content_insights):
    """Create comprehensive dashboard combining both analysis tracks."""
    print("\n📊 Creating Combined Insights Dashboard")

    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)

    # 1. Engine type comparison
    ax1 = fig.add_subplot(gs[0, 0])
    engine_stats = df.groupby('Engine')['Included'].agg(['count', 'mean']).round(3)

    colors = ['#4285F4', '#FF5722', '#9C27B0']  # Google, Bing, Perplexity colors
    bars = ax1.bar(engine_stats.index, engine_stats['mean'], color=colors, alpha=0.8)
    ax1.set_title('Inclusion Rates by Engine', fontweight='bold')
    ax1.set_ylabel('Inclusion Rate')
    ax1.tick_params(axis='x', rotation=45)

    # Add count labels
    for bar, (engine, row) in zip(bars, engine_stats.iterrows()):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{row["mean"]:.1%}\n(n={int(row["count"])})', ha='center', va='bottom', fontsize=9)

    # 2. Query type performance matrix
    ax2 = fig.add_subplot(gs[0, 1])
    query_matrix_data = []

    # Prepare data for heatmap
    trad_perf = traditional_seo.groupby('Query_Category')['Included'].mean()
    ai_perf = ai_citations.groupby('Query_Category')['Included'].mean()

    common_categories = set(trad_perf.index).intersection(set(ai_perf.index))
    matrix_data = pd.DataFrame({
        'Traditional SEO': [trad_perf.get(cat, 0) for cat in common_categories],
        'AI Citations': [ai_perf.get(cat, 0) for cat in common_categories]
    }, index=list(common_categories))

    if len(matrix_data) > 0:
        sns.heatmap(matrix_data.T, annot=True, fmt='.2f', cmap='RdYlBu_r', ax=ax2)
        ax2.set_title('Query Performance Matrix', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No Common\nCategories', ha='center', va='center',
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Query Performance Matrix', fontweight='bold')

    # 3. Cross-engine query comparison (if available)
    ax3 = fig.add_subplot(gs[0, 2])
    if len(comparison_df) > 0:
        # Scatter plot of traditional vs AI performance
        ax3.scatter(comparison_df['traditional_inclusion'], comparison_df['ai_inclusion'], alpha=0.6)
        ax3.plot([0, 1], [0, 1], 'r--', alpha=0.5)  # Equal performance line
        ax3.set_xlabel('Traditional SEO Inclusion')
        ax3.set_ylabel('AI Citation Inclusion')
        ax3.set_title('Cross-Engine Performance', fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No Common Queries\nfor Comparison', ha='center', va='center',
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Cross-Engine Performance', fontweight='bold')

    # 4. Content feature comparison
    ax4 = fig.add_subplot(gs[1, :])
    if content_insights:
        features = list(content_insights.keys())
        x = np.arange(len(features))
        width = 0.35

        trad_included = [content_insights[f]['traditional_included_mean'] for f in features]
        ai_included = [content_insights[f]['ai_included_mean'] for f in features]

        ax4.bar(x - width/2, trad_included, width, label='Traditional SEO (Included)', alpha=0.8, color='skyblue')
        ax4.bar(x + width/2, ai_included, width, label='AI Citations (Included)', alpha=0.8, color='lightcoral')

        ax4.set_xlabel('Content Features')
        ax4.set_ylabel('Average Value')
        ax4.set_title('Content Characteristics: Traditional SEO vs AI Citations', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(features, rotation=45, ha='right')
        ax4.legend()

    # 5. Dataset composition
    ax5 = fig.add_subplot(gs[2, 0])
    composition_data = df['Engine'].value_counts()
    ax5.pie(composition_data.values, labels=composition_data.index, autopct='%1.1f%%', colors=colors)
    ax5.set_title('Dataset Composition', fontweight='bold')

    # 6. Inclusion rate distribution
    ax6 = fig.add_subplot(gs[2, 1])
    inclusion_by_engine = []
    engine_labels = []

    for engine in df['Engine'].unique():
        engine_data = df[df['Engine'] == engine]['Included']
        inclusion_by_engine.append(engine_data.values)
        engine_labels.append(engine)

    ax6.boxplot(inclusion_by_engine, labels=engine_labels)
    ax6.set_title('Inclusion Rate Distribution', fontweight='bold')
    ax6.set_ylabel('Inclusion (0 or 1)')
    ax6.tick_params(axis='x', rotation=45)

    # 7. Data quality assessment
    ax7 = fig.add_subplot(gs[2, 2])
    engine_inclusion_rates = df.groupby('Engine')['Included'].mean()

    # Color code by data quality
    colors_quality = []
    for engine, rate in engine_inclusion_rates.items():
        if rate > 0.95:
            colors_quality.append('red')  # Suspiciously high
        elif rate < 0.05:
            colors_quality.append('orange')  # Suspiciously low
        else:
            colors_quality.append('green')  # Reasonable

    bars = ax7.bar(engine_inclusion_rates.index, engine_inclusion_rates.values, color=colors_quality, alpha=0.7)
    ax7.set_title('Data Quality Assessment', fontweight='bold')
    ax7.set_ylabel('Inclusion Rate')
    ax7.tick_params(axis='x', rotation=45)

    # Add quality indicators
    ax7.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='Suspiciously High')
    ax7.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='Suspiciously Low')
    ax7.legend()

    # 8. Summary and recommendations
    ax8 = fig.add_subplot(gs[3, :])
    ax8.axis('off')

    # Calculate key metrics
    total_records = len(df)
    traditional_records = len(traditional_seo)
    ai_records = len(ai_citations)
    unique_queries = df['Query_Name'].nunique()
    common_queries = len(comparison_df) if len(comparison_df) > 0 else 0

    summary_text = f"""
    🔄 COMBINED INSIGHTS: TRADITIONAL SEO + AI CITATION ANALYSIS

    📊 DATASET OVERVIEW:
    • Total Records: {total_records:,}
    • Traditional SEO Track (Google AI + Bing AI): {traditional_records:,} records
    • AI Citation Track (Perplexity): {ai_records:,} records
    • Unique Queries Analyzed: {unique_queries}
    • Common Queries for Cross-Comparison: {common_queries}

    🎯 KEY STRATEGIC INSIGHTS:
    • Different optimization strategies needed for traditional search vs AI systems
    • Traditional SEO principles still apply but require AI-specific adaptations
    • Query type performance varies significantly between systems
    • Content optimization must balance both traditional and AI-first approaches

    ⚠️ DATA QUALITY CONCERNS:
    • Perplexity shows {ai_citations['Included'].mean():.1%} inclusion rate (likely web scraping issue)
    • Google AI shows {traditional_seo[traditional_seo['Engine'] == 'Google AI']['Included'].mean():.1%} inclusion rate (selector issue)
    • Only Bing AI shows reasonable inclusion patterns

    💡 RECOMMENDATIONS:
    1. Implement dual-track content optimization strategy
    2. Fix data collection issues (Perplexity API, Google AI selectors)
    3. Focus on query types that perform well in both systems
    4. Develop AI-specific content quality metrics
    5. Monitor both traditional rankings and AI citation patterns

    🔬 RESEARCH IMPLICATIONS:
    This analysis demonstrates the need for separate methodologies when studying
    traditional search engines vs AI-first answer systems. Future research should
    maintain this distinction while looking for universal optimization principles.
    """

    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))

    plt.suptitle('Combined Insights: Traditional SEO + AI Citation Analysis', fontsize=18, fontweight='bold')

    # Save
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/combined_insights_analysis.png", dpi=300, bbox_inches='tight')
    print("✅ Combined insights dashboard saved to: plots/combined_insights_analysis.png")
    plt.close()

def main():
    """Main analysis execution."""
    print("🔄 Combined Insights Analysis: Traditional SEO + AI Citations")
    print("="*60)

    # Load and segment data
    df, traditional_seo, ai_citations = load_and_segment_data()

    # Cross-engine query analysis
    comparison_df = cross_engine_query_analysis(traditional_seo, ai_citations)

    # Content optimization insights
    content_insights = analyze_content_optimization_insights(traditional_seo, ai_citations)

    # Universal principles
    universal_principles = identify_universal_principles(traditional_seo, ai_citations)

    # Create combined dashboard
    create_combined_insights_dashboard(df, traditional_seo, ai_citations, comparison_df, content_insights)

    print(f"\n✅ Combined Insights Analysis Complete!")
    print(f"📁 Results saved to plots/combined_insights_analysis.png")
    print(f"🎯 Strategic insights for dual-track optimization approach")

if __name__ == "__main__":
    main()