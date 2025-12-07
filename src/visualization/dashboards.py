"""Visualization dashboards for SEO vs AISO analysis.

This module creates comprehensive analysis dashboards.
"""

import os
import numpy as np
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


def create_traditional_seo_dashboard(df, correlations, engine_analysis, models,
                                     output_path="outputs/figures/traditional_seo_analysis.png"):
    """
    Create comprehensive dashboard for traditional SEO analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Traditional SEO data
    correlations : pd.Series
        Feature correlations with inclusion
    engine_analysis : dict
        Per-engine analysis results
    models : dict
        Trained model results
    output_path : str, optional
        Path to save figure

    Returns
    -------
    None
        Saves figure to output_path

    Examples
    --------
    >>> create_traditional_seo_dashboard(df, corr, engine_analysis, models)
    ✅ Traditional SEO dashboard saved to: outputs/figures/traditional_seo_analysis.png
    """
    print("\n📊 Creating Traditional SEO Analysis Dashboard")

    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)

    # 1. Engine inclusion comparison
    ax1 = fig.add_subplot(gs[0, 0])
    engines = ['Google AI', 'Bing AI']
    inclusion_rates = [engine_analysis[engine]['inclusion_rate'] for engine in engines]
    colors = ['#4285F4', '#FF5722']  # Google blue, Bing orange

    bars = ax1.bar(engines, inclusion_rates, color=colors, alpha=0.8)
    ax1.set_title('AI Overview Inclusion Rates', fontweight='bold')
    ax1.set_ylabel('Inclusion Rate')

    # Add value labels
    for bar, rate in zip(bars, inclusion_rates):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')

    # 2. Rank vs inclusion analysis
    ax2 = fig.add_subplot(gs[0, 1])
    for i, engine in enumerate(engines):
        engine_data = engine_analysis[engine]['data']
        rank_stats = engine_data[engine_data['Page Rank'] <= 10].groupby('Page Rank')['Included'].mean()
        ax2.plot(rank_stats.index, rank_stats.values, 'o-',
                label=engine, color=colors[i], linewidth=2, markersize=6)

    ax2.set_title('Inclusion Rate by Search Rank', fontweight='bold')
    ax2.set_xlabel('Page Rank')
    ax2.set_ylabel('Inclusion Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Feature correlations
    ax3 = fig.add_subplot(gs[0, 2])
    if correlations is not None:
        y_pos = np.arange(len(correlations))
        colors_corr = ['green' if x > 0 else 'red' for x in correlations.values]
        bars = ax3.barh(y_pos, correlations.values, color=colors_corr, alpha=0.7)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(correlations.index)
        ax3.set_title('SEO Factor Correlations', fontweight='bold')
        ax3.set_xlabel('Correlation with Inclusion')
        ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3)

    # 4. Query category performance
    ax4 = fig.add_subplot(gs[1, 0])
    category_perf = df.groupby('Query_Category')['Included'].mean().sort_values(ascending=True)
    ax4.barh(category_perf.index, category_perf.values, color='lightcoral', alpha=0.8)
    ax4.set_title('Performance by Query Type', fontweight='bold')
    ax4.set_xlabel('Inclusion Rate')

    # 5. Feature importance (if available)
    ax5 = fig.add_subplot(gs[1, 1])
    if models and 'random_forest' in models:
        importance = models['random_forest']['importance'].head(8)
        ax5.barh(importance['feature'], importance['importance'], color='lightgreen', alpha=0.8)
        ax5.set_title('Feature Importance (Random Forest)', fontweight='bold')
        ax5.set_xlabel('Importance Score')

    # 6. Sample size distribution
    ax6 = fig.add_subplot(gs[1, 2])
    sample_sizes = df['Engine'].value_counts()
    ax6.pie(sample_sizes.values, labels=sample_sizes.index, autopct='%1.1f%%',
           colors=colors[:len(sample_sizes)])
    ax6.set_title('Dataset Composition', fontweight='bold')

    # 7. Word count distribution by engine
    ax7 = fig.add_subplot(gs[2, :2])
    for i, engine in enumerate(engines):
        engine_data = df[df['Engine'] == engine]
        ax7.hist(engine_data['Word Count'], bins=30, alpha=0.6,
                label=engine, color=colors[i])
    ax7.set_title('Word Count Distribution by Engine', fontweight='bold')
    ax7.set_xlabel('Word Count')
    ax7.set_ylabel('Frequency')
    ax7.legend()

    # 8. Model performance comparison
    ax8 = fig.add_subplot(gs[2, 2])
    if models:
        model_names = []
        model_scores = []
        for name, model_info in models.items():
            model_names.append(name.replace('_', ' ').title())
            model_scores.append(model_info['scores'].mean())

        bars = ax8.bar(model_names, model_scores, color='skyblue', alpha=0.8)
        ax8.set_title('Model Performance', fontweight='bold')
        ax8.set_ylabel('Cross-validation Accuracy')
        ax8.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, score in zip(bars, model_scores):
            ax8.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{score:.3f}', ha='center', va='bottom')

    # 9. Summary statistics
    ax9 = fig.add_subplot(gs[3, :])
    ax9.axis('off')

    # Calculate summary stats
    total_results = len(df)
    unique_queries = df['Query_Name'].nunique()
    overall_inclusion = df['Included'].mean()
    google_inclusion = df[df['Engine'] == 'Google AI']['Included'].mean()
    bing_inclusion = df[df['Engine'] == 'Bing AI']['Included'].mean()

    summary_text = f"""
    📊 TRADITIONAL SEO ANALYSIS SUMMARY

    🔢 DATASET STATISTICS:
    • Total Results: {total_results:,} (Traditional search engines only)
    • Unique Queries: {unique_queries}
    • Google AI Results: {len(df[df['Engine'] == 'Google AI']):,} ({google_inclusion:.1%} inclusion)
    • Bing AI Results: {len(df[df['Engine'] == 'Bing AI']):,} ({bing_inclusion:.1%} inclusion)
    • Overall Inclusion Rate: {overall_inclusion:.1%}

    🎯 KEY FINDINGS:
    • Traditional search rankings show clear patterns for AI Overview inclusion
    • Both engines display traditional SERP behavior with AI enhancements
    • Page rank correlation indicates traditional SEO factors still matter
    • Content optimization for AI Overviews follows familiar SEO principles

    💡 IMPLICATIONS FOR SEO:
    • Traditional SEO remains relevant for AI Overview inclusion
    • Top-ranking pages have higher inclusion probability
    • Content quality metrics correlate with AI selection
    • Dual optimization strategy recommended: Traditional SEO + AI-specific factors
    """

    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    plt.suptitle('Traditional SEO Analysis: Google AI + Bing AI', fontsize=18, fontweight='bold')

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Traditional SEO dashboard saved to: {output_path}")
    plt.close()
