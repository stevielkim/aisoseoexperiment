#!/usr/bin/env python3
"""
Data-Quality Aware Analysis for SEO vs AISO Study
Addresses the critical data quality issues identified in the comprehensive report.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("RdYlBu_r")

def load_and_filter_data():
    """Load data and apply quality filters."""
    print("🔍 Loading and filtering data for quality analysis...")

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
        elif query.startswith(('benefits of', 'advantages')):
            return 'Benefits'
        else:
            return 'Other'

    df['Query_Category'] = df['Query_Name'].apply(categorize_query)

    # Print data quality assessment
    print(f"📊 Total dataset: {len(df):,} results")
    print(f"🔍 Unique queries: {df['Query_Name'].nunique()}")
    print(f"🏭 Engines: {', '.join(df['Engine'].unique())}")

    # Identify problematic engines
    engine_rates = df.groupby('Engine')['Included'].agg(['count', 'sum', 'mean'])
    engine_rates['inclusion_rate'] = engine_rates['mean']

    print("\n🚨 ENGINE QUALITY ASSESSMENT:")
    problematic_engines = []
    clean_engines = []

    for engine, row in engine_rates.iterrows():
        rate = row['inclusion_rate']
        count = row['count']
        included = row['sum']

        print(f"  {engine}: {included:,}/{count:,} ({rate:.1%})")

        if rate > 0.95:
            print(f"    ⚠️ CRITICAL: {engine} inclusion too high - likely parsing error")
            problematic_engines.append(engine)
        elif rate < 0.05:
            print(f"    ⚠️ CRITICAL: {engine} inclusion too low - may not be working")
            problematic_engines.append(engine)
        else:
            print(f"    ✅ {engine} inclusion rate is reasonable")
            clean_engines.append(engine)

    return df, problematic_engines, clean_engines

def create_data_quality_dashboard(df, problematic_engines, clean_engines):
    """Create visualizations focusing on data quality issues."""

    print("\n📊 Creating data quality-focused visualizations...")

    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)

    # 1. Engine Inclusion Rates with Quality Flags
    ax1 = fig.add_subplot(gs[0, 0])
    engine_stats = df.groupby('Engine')['Included'].mean()
    colors = ['red' if engine in problematic_engines else 'green' for engine in engine_stats.index]
    bars = ax1.bar(engine_stats.index, engine_stats.values, color=colors, alpha=0.7)
    ax1.set_title('Engine Inclusion Rates\n(Red = Data Quality Issues)', fontweight='bold')
    ax1.set_ylabel('Inclusion Rate')
    ax1.tick_params(axis='x', rotation=45)

    # Add value labels and quality flags
    for bar, engine in zip(bars, engine_stats.index):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom')
        if engine in problematic_engines:
            ax1.text(bar.get_x() + bar.get_width()/2., height/2,
                    '⚠️', ha='center', va='center', fontsize=20)

    # 2. Clean Engine Comparison (Exclude Problematic)
    ax2 = fig.add_subplot(gs[0, 1])
    if clean_engines:
        clean_df = df[df['Engine'].isin(clean_engines)]
        clean_stats = clean_df.groupby('Engine')['Included'].mean()
        ax2.bar(clean_stats.index, clean_stats.values, color='lightgreen', alpha=0.8)
        ax2.set_title('Clean Engines Only\n(Reliable Data)', fontweight='bold')
        ax2.set_ylabel('Inclusion Rate')
        for i, v in enumerate(clean_stats.values):
            ax2.text(i, v + 0.01, f'{v:.1%}', ha='center', va='bottom')
    else:
        ax2.text(0.5, 0.5, 'No Clean Engines\nFound', ha='center', va='center',
                transform=ax2.transAxes, fontsize=14, color='red')
        ax2.set_title('Clean Engines Only', fontweight='bold')

    # 3. Query Category Performance (All Engines)
    ax3 = fig.add_subplot(gs[0, 2])
    query_perf = df.groupby('Query_Category')['Included'].mean().sort_values(ascending=True)
    ax3.barh(query_perf.index, query_perf.values, color='lightcoral')
    ax3.set_title('Query Type Performance\n(All Engines)', fontweight='bold')
    ax3.set_xlabel('Inclusion Rate')

    # 4. Engine Sample Size Distribution
    ax4 = fig.add_subplot(gs[1, 0])
    sample_sizes = df['Engine'].value_counts()
    ax4.pie(sample_sizes.values, labels=sample_sizes.index, autopct='%1.1f%%')
    ax4.set_title('Dataset Composition\n(Sample Sizes)', fontweight='bold')

    # 5. Inclusion by Page Rank (Clean Engines Only)
    ax5 = fig.add_subplot(gs[1, 1])
    if clean_engines:
        clean_df = df[df['Engine'].isin(clean_engines)]
        rank_data = clean_df[clean_df['Page Rank'] <= 10].groupby('Page Rank')['Included'].mean()
        ax5.plot(rank_data.index, rank_data.values, 'o-', color='green', linewidth=2)
        ax5.set_title('Rank vs Inclusion\n(Clean Engines Only)', fontweight='bold')
        ax5.set_xlabel('Page Rank')
        ax5.set_ylabel('Inclusion Rate')
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'No Clean Engine\nData Available', ha='center', va='center',
                transform=ax5.transAxes, fontsize=12, color='red')

    # 6. Feature Correlation Heatmap (Clean Engines)
    ax6 = fig.add_subplot(gs[1, 2])
    feature_cols = ['Word Count', 'H1 Count', 'H2 Count', 'MetaDesc Length', 'Page Rank', 'Included']
    available_cols = [col for col in feature_cols if col in df.columns]

    if len(available_cols) >= 3 and clean_engines:
        clean_df = df[df['Engine'].isin(clean_engines)]
        corr_data = clean_df[available_cols].corr()
        mask = np.triu(np.ones_like(corr_data, dtype=bool))
        sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, mask=mask, ax=ax6)
        ax6.set_title('Feature Correlations\n(Clean Data Only)', fontweight='bold')
    else:
        ax6.text(0.5, 0.5, 'Insufficient Data\nfor Correlation', ha='center', va='center',
                transform=ax6.transAxes, fontsize=12, color='red')

    # 7. Query Performance Distribution
    ax7 = fig.add_subplot(gs[2, :])
    query_stats = df.groupby('Query_Name').agg({
        'Included': ['mean', 'count', 'sum']
    }).round(3)
    query_stats.columns = ['inclusion_rate', 'total_results', 'included_count']
    query_stats = query_stats[query_stats['total_results'] >= 3]  # At least 3 engines
    query_stats = query_stats.sort_values('inclusion_rate', ascending=True)

    # Show top 20 and bottom 20 queries
    top_queries = query_stats.tail(20)
    bottom_queries = query_stats.head(20)
    combined = pd.concat([bottom_queries, top_queries])

    bars = ax7.barh(range(len(combined)), combined['inclusion_rate'])
    ax7.set_yticks(range(len(combined)))
    ax7.set_yticklabels([q[:50] + '...' if len(q) > 50 else q for q in combined.index], fontsize=8)
    ax7.set_xlabel('Inclusion Rate')
    ax7.set_title('Query Performance Spectrum (Bottom 20 → Top 20)', fontweight='bold')
    ax7.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='50% threshold')
    ax7.legend()

    # 8. Data Quality Summary Box
    ax8 = fig.add_subplot(gs[3, :])
    ax8.axis('off')

    summary_text = f"""
    📊 DATA QUALITY SUMMARY

    ✅ RELIABLE ENGINES: {', '.join(clean_engines) if clean_engines else 'None identified'}
    ⚠️ PROBLEMATIC ENGINES: {', '.join(problematic_engines) if problematic_engines else 'None identified'}

    🔢 DATASET STATS:
    • Total Results: {len(df):,}
    • Unique Queries: {df['Query_Name'].nunique()}
    • Query Categories: {', '.join(df['Query_Category'].unique())}
    • Overall Inclusion Rate: {df['Included'].mean():.1%}

    🚨 CRITICAL ISSUES IDENTIFIED:
    • Perplexity: 97.0% inclusion (likely web scraping parsing error)
    • Google AI: 0.3% inclusion (AI Overviews may not be appearing)
    • Only Bing AI shows reasonable inclusion rates (8.7%)

    💡 IMMEDIATE ACTIONS NEEDED:
    1. Implement Perplexity API for accurate citation data
    2. Debug Google AI selectors for AI Overview detection
    3. Validate Bing AI results for consistency
    4. Re-run analysis with clean data
    """

    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    plt.suptitle('SEO vs AISO: Data Quality-Focused Analysis', fontsize=18, fontweight='bold')

    # Save the plot
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/data_quality_analysis.png", dpi=300, bbox_inches='tight')
    print("✅ Data quality analysis saved to: plots/data_quality_analysis.png")
    plt.close()

    return query_stats

def generate_actionable_insights(df, problematic_engines, clean_engines):
    """Generate specific, actionable insights based on the analysis."""

    print("\n🎯 GENERATING ACTIONABLE INSIGHTS")
    print("="*60)

    # Only analyze clean engines for reliable insights
    if clean_engines:
        clean_df = df[df['Engine'].isin(clean_engines)]
        print(f"📊 Analysis based on {len(clean_df):,} results from reliable engines: {', '.join(clean_engines)}")

        # Best performing query types (clean data)
        clean_query_perf = clean_df.groupby('Query_Category')['Included'].agg(['mean', 'count'])
        clean_query_perf = clean_query_perf[clean_query_perf['count'] >= 10]  # Sufficient sample size

        if not clean_query_perf.empty:
            best_category = clean_query_perf['mean'].idxmax()
            worst_category = clean_query_perf['mean'].idxmin()

            print(f"\n🏆 RELIABLE INSIGHTS FROM CLEAN DATA:")
            print(f"  • Best performing query type: {best_category} ({clean_query_perf.loc[best_category, 'mean']:.1%})")
            print(f"  • Worst performing query type: {worst_category} ({clean_query_perf.loc[worst_category, 'mean']:.1%})")

            # Feature correlations from clean data
            feature_cols = ['Word Count', 'H1 Count', 'H2 Count', 'MetaDesc Length', 'Page Rank']
            available_features = [col for col in feature_cols if col in clean_df.columns]

            if len(available_features) >= 3:
                feature_corrs = clean_df[available_features + ['Included']].corr()['Included'].abs().sort_values(ascending=False)
                print(f"\n🔗 TOP FACTORS FOR INCLUSION (Clean Data):")
                for feature, corr in feature_corrs.head(5).items():
                    if feature != 'Included':
                        print(f"  • {feature}: {corr:.3f} correlation")

        # Rank analysis
        if 'Page Rank' in clean_df.columns:
            rank_analysis = clean_df[clean_df['Page Rank'] <= 10].groupby('Page Rank')['Included'].mean()
            if len(rank_analysis) >= 5:
                print(f"\n🏅 RANK PERFORMANCE (Clean Data):")
                print(f"  • Rank 1 inclusion: {rank_analysis.get(1, 0):.1%}")
                print(f"  • Average top 3 inclusion: {rank_analysis.head(3).mean():.1%}")
                print(f"  • Average rank 4-10 inclusion: {rank_analysis.tail(7).mean():.1%}")

    else:
        print("❌ No reliable engines identified - cannot generate trustworthy insights")
        print("   All engines show data quality issues that need to be resolved first")

    print(f"\n🚨 CRITICAL DATA QUALITY ISSUES:")
    for engine in problematic_engines:
        engine_rate = df[df['Engine'] == engine]['Included'].mean()
        if engine_rate > 0.95:
            print(f"  • {engine}: {engine_rate:.1%} inclusion - Web scraping parsing error")
            print(f"    → SOLUTION: Implement {engine} API integration")
        elif engine_rate < 0.05:
            print(f"  • {engine}: {engine_rate:.1%} inclusion - Selectors not working")
            print(f"    → SOLUTION: Update {engine} CSS selectors and parsing logic")

    print(f"\n💡 IMMEDIATE NEXT STEPS:")
    print(f"  1. Fix Perplexity data collection using API (highest priority)")
    print(f"  2. Debug Google AI selectors for AI Overview detection")
    print(f"  3. Validate Bing AI results (currently most reliable)")
    print(f"  4. Re-run complete analysis after data fixes")
    print(f"  5. Only then draw conclusions about SEO vs AISO effectiveness")

def main():
    """Main execution function."""
    print("🔍 SEO vs AISO: Data Quality-Focused Analysis")
    print("="*60)

    # Load and assess data quality
    df, problematic_engines, clean_engines = load_and_filter_data()

    # Create quality-focused visualizations
    query_stats = create_data_quality_dashboard(df, problematic_engines, clean_engines)

    # Generate actionable insights
    generate_actionable_insights(df, problematic_engines, clean_engines)

    print(f"\n✅ Data quality analysis complete!")
    print(f"📁 Check 'plots/data_quality_analysis.png' for comprehensive visualization")
    print(f"🎯 Focus on resolving data quality issues before drawing research conclusions")

if __name__ == "__main__":
    main()