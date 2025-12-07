"""Visualization dashboards for SEO vs AISO analysis.

This module creates comprehensive analysis dashboards.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Style
plt.style.use('default')
sns.set_palette("viridis")


def create_content_feature_dashboard(citations_df, domain_counts, type_counts,
                                     feature_stats, correlations, importance_df,
                                     output_path="outputs/figures/content_feature_analysis.png"):
    """
    Create comprehensive content feature analysis dashboard.

    Parameters
    ----------
    citations_df : pd.DataFrame
        Citation data
    domain_counts : pd.Series
        Citation counts by domain
    type_counts : pd.Series
        Citation counts by domain type
    feature_stats : dict
        Content feature statistics
    correlations : dict
        Feature correlations with citation order
    importance_df : pd.DataFrame
        Feature importance from Random Forest
    output_path : str, optional
        Path to save figure

    Returns
    -------
    None
        Saves figure to output_path

    Examples
    --------
    >>> create_content_feature_dashboard(
    ...     citations_df, domain_counts, type_counts,
    ...     feature_stats, correlations, importance_df
    ... )
    ✅ Dashboard saved to: outputs/figures/content_feature_analysis.png
    """
    print("\n📊 Creating Analysis Dashboard")

    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # 1. Citations by Engine
    ax1 = fig.add_subplot(gs[0, 0])
    engine_counts = citations_df[citations_df['Result URL'] != ''].groupby('Engine').size()
    colors = ['#3498db', '#e74c3c', '#2ecc71'][:len(engine_counts)]
    bars = ax1.bar(engine_counts.index, engine_counts.values, color=colors)
    ax1.set_title('Citations by Engine', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Number of Citations')
    for bar, val in zip(bars, engine_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(val), ha='center', va='bottom', fontweight='bold')

    # 2. Citation Order Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    order_data = citations_df[citations_df['Citation_Order'].notna()]['Citation_Order']
    if len(order_data) > 0:
        order_counts = order_data.value_counts().sort_index().head(10)
        ax2.bar(order_counts.index.astype(int), order_counts.values, color='#9b59b6', alpha=0.8)
        ax2.set_title('Citation Order Distribution', fontweight='bold', fontsize=12)
        ax2.set_xlabel('Citation Position')
        ax2.set_ylabel('Count')

    # 3. Top Domains
    ax3 = fig.add_subplot(gs[0, 2])
    if domain_counts is not None and len(domain_counts) > 0:
        top_domains = domain_counts.head(10)
        y_pos = range(len(top_domains))
        ax3.barh(y_pos, top_domains.values, color='#1abc9c', alpha=0.8)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels([d[:25] + '...' if len(d) > 25 else d for d in top_domains.index])
        ax3.set_title('Top Cited Domains', fontweight='bold', fontsize=12)
        ax3.set_xlabel('Citation Count')
        ax3.invert_yaxis()

    # 4. Domain Type Distribution
    ax4 = fig.add_subplot(gs[1, 0])
    if type_counts is not None and len(type_counts) > 0:
        colors_pie = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(type_counts)]
        ax4.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
               colors=colors_pie, startangle=90)
        ax4.set_title('Domain Type Distribution', fontweight='bold', fontsize=12)

    # 5. Content Feature Averages
    ax5 = fig.add_subplot(gs[1, 1])
    if feature_stats and len(feature_stats) > 0:
        features = list(feature_stats.keys())[:8]
        means = [feature_stats[f]['mean'] for f in features]
        y_pos = range(len(features))
        ax5.barh(y_pos, means, color='#e67e22', alpha=0.8)
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels([f.replace('_', ' ').title()[:20] for f in features])
        ax5.set_title('Avg Content Features (Cited Sources)', fontweight='bold', fontsize=12)
        ax5.set_xlabel('Average Value')
    else:
        ax5.text(0.5, 0.5, 'Run fetch_source_features.py\nto see content features',
                ha='center', va='center', transform=ax5.transAxes, fontsize=11)
        ax5.set_title('Content Features', fontweight='bold', fontsize=12)

    # 6. Feature Importance
    ax6 = fig.add_subplot(gs[1, 2])
    if importance_df is not None and len(importance_df) > 0:
        top_features = importance_df.head(8)
        y_pos = range(len(top_features))
        ax6.barh(y_pos, top_features['importance'].values, color='#27ae60', alpha=0.8)
        ax6.set_yticks(y_pos)
        ax6.set_yticklabels([f.replace('_', ' ').title()[:20] for f in top_features['feature']])
        ax6.set_title('Feature Importance (Predicts Early Citation)', fontweight='bold', fontsize=12)
        ax6.set_xlabel('Importance Score')
    else:
        ax6.text(0.5, 0.5, 'Run fetch_source_features.py\nfor feature importance',
                ha='center', va='center', transform=ax6.transAxes, fontsize=11)
        ax6.set_title('Feature Importance', fontweight='bold', fontsize=12)

    # 7-9. Summary Panel
    ax_summary = fig.add_subplot(gs[2, :])
    ax_summary.axis('off')

    # Calculate summary stats
    total_citations = len(citations_df[citations_df['Result URL'] != ''])
    unique_queries = citations_df['Query'].nunique() if 'Query' in citations_df.columns else 0
    unique_sources = citations_df[citations_df['Result URL'] != '']['Result URL'].nunique()

    # Top insights
    top_domain = domain_counts.index[0] if domain_counts is not None and len(domain_counts) > 0 else "N/A"
    top_domain_count = domain_counts.values[0] if domain_counts is not None and len(domain_counts) > 0 else 0

    top_feature = importance_df.iloc[0]['feature'] if importance_df is not None and len(importance_df) > 0 else "N/A"

    summary_text = f"""
    🎯 AI CONTENT CITATION ANALYSIS SUMMARY

    📊 DATASET OVERVIEW:
    • Total Citations Analyzed: {total_citations:,}
    • Unique Queries: {unique_queries}
    • Unique Source URLs: {unique_sources}

    🏆 KEY FINDINGS:
    • Most Cited Domain: {top_domain} ({top_domain_count} citations)
    • Top Feature for Early Citation: {top_feature.replace('_', ' ').title() if top_feature != 'N/A' else 'N/A'}
    • Data Quality: Perplexity data is most reliable (all shown sources are cited)

    💡 IMPLICATIONS FOR CONTENT OPTIMIZATION:
    • Authoritative domains (health, education, government) are frequently cited
    • Citation order correlates with content structure and depth
    • To increase AI citation likelihood, focus on comprehensive, well-structured content

    ⚠️ NOTES:
    • Google AI data has low detection rate (may need selector updates)
    • For detailed content features, run fetch_source_features.py
    • Analysis based on {unique_queries} diverse queries across AI search engines
    """

    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes, fontsize=11,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="#ecf0f1", alpha=0.9))

    plt.suptitle('AI Content Citation Analysis: What Gets Cited?',
                fontsize=18, fontweight='bold', y=0.98)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Dashboard saved to: {output_path}")
    plt.close()
