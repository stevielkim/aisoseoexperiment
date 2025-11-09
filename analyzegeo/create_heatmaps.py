#!/usr/bin/env python3
"""
Create focused heatmaps for SEO vs AISO analysis.
Addresses the data quality issues and highlights key patterns.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
plt.style.use('default')
sns.set_palette("RdYlBu_r")

def load_and_prepare_data():
    """Load and prepare data for heatmap analysis."""
    df = pd.read_csv('ai_serp_analysis.csv')
    df['Included'] = df['Included'].astype(int)

    # Extract query info
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
    return df

def create_engine_query_heatmap(df, save_path="plots/engine_query_heatmap.png"):
    """Create heatmap showing inclusion rates by engine and query category."""
    # Calculate inclusion rates
    heatmap_data = df.groupby(['Query_Category', 'Engine'])['Included'].mean().unstack(fill_value=0)

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data,
                annot=True,
                fmt='.2f',
                cmap='RdYlBu_r',
                center=0.5,
                square=True,
                linewidths=0.5,
                cbar_kws={'label': 'Inclusion Rate'})

    plt.title('Inclusion Rates: Query Category vs Engine', fontsize=14, fontweight='bold')
    plt.xlabel('Engine', fontweight='bold')
    plt.ylabel('Query Category', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Engine-Query heatmap saved to: {save_path}")
    plt.show()

    return heatmap_data

def create_feature_correlation_heatmap(df, save_path="plots/feature_correlation_heatmap.png"):
    """Create correlation heatmap of SEO features."""
    # Select numeric features for correlation
    feature_cols = ['Word Count', 'H1 Count', 'H2 Count', 'H3 Count',
                   'MetaDesc Length', 'List Count', 'Image Count',
                   'Snippet Length', 'Page Rank', 'Included']

    # Filter to columns that exist
    available_cols = [col for col in feature_cols if col in df.columns]

    if len(available_cols) < 3:
        print("❌ Not enough numeric columns for correlation heatmap")
        return None

    # Calculate correlations
    correlation_data = df[available_cols].corr()

    # Create the plot
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(correlation_data, dtype=bool))
    sns.heatmap(correlation_data,
                annot=True,
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                square=True,
                mask=mask,
                linewidths=0.5,
                cbar_kws={'label': 'Correlation Coefficient'})

    plt.title('SEO Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Feature correlation heatmap saved to: {save_path}")
    plt.show()

    return correlation_data

def create_rank_performance_heatmap(df, save_path="plots/rank_performance_heatmap.png"):
    """Create heatmap showing inclusion by rank and engine."""
    # Group by rank (1-10) and engine
    rank_data = df[df['Page Rank'] <= 10].groupby(['Page Rank', 'Engine'])['Included'].mean().unstack(fill_value=0)

    plt.figure(figsize=(8, 10))
    sns.heatmap(rank_data,
                annot=True,
                fmt='.2f',
                cmap='RdYlBu_r',
                center=0.5,
                linewidths=0.5,
                cbar_kws={'label': 'Inclusion Rate'})

    plt.title('Inclusion Rate by Page Rank and Engine', fontsize=14, fontweight='bold')
    plt.xlabel('Engine', fontweight='bold')
    plt.ylabel('Page Rank', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Rank performance heatmap saved to: {save_path}")
    plt.show()

    return rank_data

def create_query_difficulty_heatmap(df, save_path="plots/query_difficulty_heatmap.png"):
    """Create heatmap showing which queries are hardest/easiest to rank for."""
    # Calculate query-level statistics
    query_stats = df.groupby('Query_Name').agg({
        'Included': ['mean', 'count'],
        'Page Rank': 'mean',
        'Word Count': 'mean'
    }).round(2)

    # Flatten column names
    query_stats.columns = ['_'.join(col).strip() for col in query_stats.columns]

    # Filter queries with enough data points
    query_stats_filtered = query_stats[query_stats['Included_count'] >= 6]

    if len(query_stats_filtered) < 10:
        print("❌ Not enough queries with sufficient data points")
        return None

    # Select top and bottom performers for visualization
    top_queries = query_stats_filtered.nlargest(15, 'Included_mean')
    bottom_queries = query_stats_filtered.nsmallest(15, 'Included_mean')
    combined_queries = pd.concat([top_queries, bottom_queries]).drop_duplicates()

    # Prepare data for heatmap
    heatmap_cols = ['Included_mean', 'Page Rank_mean', 'Word Count_mean']
    heatmap_data = combined_queries[heatmap_cols].T

    # Normalize for better visualization
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    heatmap_normalized = pd.DataFrame(
        scaler.fit_transform(heatmap_data.T).T,
        index=['Inclusion Rate', 'Avg Page Rank', 'Avg Word Count'],
        columns=heatmap_data.columns
    )

    plt.figure(figsize=(16, 6))
    sns.heatmap(heatmap_normalized,
                annot=False,
                cmap='RdYlBu_r',
                center=0,
                linewidths=0.5,
                cbar_kws={'label': 'Normalized Score'})

    plt.title('Query Difficulty Analysis (Top & Bottom Performers)', fontsize=14, fontweight='bold')
    plt.xlabel('Query', fontweight='bold')
    plt.ylabel('Metric', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Query difficulty heatmap saved to: {save_path}")
    plt.show()

    return combined_queries

def generate_insights(engine_query_data, correlation_data, rank_data):
    """Generate key insights from the heatmap analysis."""
    print("\n🎯 KEY INSIGHTS FROM HEATMAP ANALYSIS")
    print("="*60)

    # Engine insights
    print("📊 ENGINE PERFORMANCE:")
    best_engine = engine_query_data.mean(axis=0).idxmax()
    worst_engine = engine_query_data.mean(axis=0).idxmin()
    print(f"  • Best performing engine: {best_engine}")
    print(f"  • Worst performing engine: {worst_engine}")

    # Query type insights
    print(f"\n📝 QUERY TYPE PERFORMANCE:")
    best_query_type = engine_query_data.mean(axis=1).idxmax()
    worst_query_type = engine_query_data.mean(axis=1).idxmin()
    print(f"  • Best performing query type: {best_query_type}")
    print(f"  • Worst performing query type: {worst_query_type}")

    # Feature correlations (if available)
    if correlation_data is not None and 'Included' in correlation_data.columns:
        print(f"\n🔗 FEATURE CORRELATIONS WITH INCLUSION:")
        inclusion_corr = correlation_data['Included'].abs().sort_values(ascending=False)[1:]  # Exclude self-correlation
        for feature, corr in inclusion_corr.head(5).items():
            print(f"  • {feature}: {corr:.3f}")

    # Rank insights
    if rank_data is not None:
        print(f"\n🏆 RANK PERFORMANCE:")
        rank_1_best = rank_data.loc[1].idxmax() if 1 in rank_data.index else "N/A"
        if rank_1_best != "N/A":
            print(f"  • Best engine for rank #1: {rank_1_best}")

def main():
    """Main execution function."""
    print("🔥 Creating Enhanced Heatmaps for SEO vs AISO Analysis")
    print("="*60)

    # Create plots directory
    os.makedirs("plots", exist_ok=True)

    # Load data
    df = load_and_prepare_data()
    print(f"📊 Loaded {len(df):,} results for analysis")

    # Create heatmaps
    engine_query_data = create_engine_query_heatmap(df)
    correlation_data = create_feature_correlation_heatmap(df)
    rank_data = create_rank_performance_heatmap(df)
    query_difficulty_data = create_query_difficulty_heatmap(df)

    # Generate insights
    generate_insights(engine_query_data, correlation_data, rank_data)

    print(f"\n✅ All heatmaps generated successfully!")
    print(f"📁 Check the 'plots/' directory for all visualizations")

if __name__ == "__main__":
    main()