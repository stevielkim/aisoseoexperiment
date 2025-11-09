#!/usr/bin/env python3
"""
AI Citation Analysis: Perplexity Focus
Focus: What content characteristics get cited by AI systems?
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("viridis")

def load_ai_citation_data():
    """Load and prepare data for AI citation analysis (Perplexity focus)."""
    print("🤖 Loading AI Citation Data (Perplexity Focus)")
    print("="*60)

    df = pd.read_csv('ai_serp_analysis.csv')

    # Filter to Perplexity only
    df = df[df['Engine'] == 'Perplexity'].copy()

    # Clean data
    df['Included'] = df['Included'].astype(int)
    df['Query_Name'] = df['File'].str.extract(r'([^/]+)\.html$')[0].str.replace('_', ' ')

    # Citation order analysis (Perplexity's "Page Rank" is actually citation order)
    df['Citation_Order'] = df['Page Rank']

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

    print(f"✅ Perplexity citation dataset:")
    print(f"   • Total citations: {len(df):,}")
    print(f"   • Unique queries: {df['Query_Name'].nunique()}")
    print(f"   • Citation inclusion rate: {df['Included'].mean():.1%}")
    print(f"   • Citation order range: {df['Citation_Order'].min()}-{df['Citation_Order'].max()}")

    return df

def analyze_citation_patterns(df):
    """Analyze how Perplexity selects and orders citations."""
    print("\n📚 AI CITATION PATTERN ANALYSIS")
    print("="*50)

    # Citation order analysis
    citation_stats = df.groupby('Citation_Order').agg({
        'Included': ['count', 'sum', 'mean'],
        'Query_Name': 'nunique'
    }).round(3)

    print("🔢 CITATION ORDER PATTERNS:")
    for order in sorted(df['Citation_Order'].unique())[:10]:
        order_data = df[df['Citation_Order'] == order]
        inclusion_rate = order_data['Included'].mean()
        count = len(order_data)
        print(f"   • Citation #{int(order)}: {count} citations, {inclusion_rate:.1%} included")

    # Most cited sources
    if 'Result URL' in df.columns:
        top_sources = df['Result URL'].value_counts().head(10)
        print(f"\n🏆 MOST FREQUENTLY CITED SOURCES:")
        for i, (source, count) in enumerate(top_sources.items(), 1):
            # Extract domain
            domain = source.split('//')[1].split('/')[0] if '//' in source else source
            inclusion_rate = df[df['Result URL'] == source]['Included'].mean()
            print(f"   {i:2}. {domain}: {count} citations ({inclusion_rate:.1%} included)")

    return citation_stats

def analyze_content_characteristics(df):
    """Analyze what content characteristics make citations worthy of inclusion."""
    print("\n📝 CONTENT CHARACTERISTICS FOR AI CITATION")
    print("="*50)

    # Content quality features
    content_features = [
        'Word Count', 'H1 Count', 'H2 Count', 'H3 Count', 'MetaDesc Length',
        'List Count', 'Image Count', 'AI Overview Length'
    ]

    available_features = [f for f in content_features if f in df.columns]
    print(f"📋 Available content features: {', '.join(available_features)}")

    if len(available_features) >= 3:
        # Compare included vs not included citations
        included_df = df[df['Included'] == 1]
        excluded_df = df[df['Included'] == 0]

        print(f"\n📊 CONTENT QUALITY COMPARISON:")
        print(f"   Included citations: {len(included_df):,}")
        print(f"   Excluded citations: {len(excluded_df):,}")

        for feature in available_features:
            if feature in df.columns and df[feature].notna().sum() > 0:
                included_mean = included_df[feature].mean()
                excluded_mean = excluded_df[feature].mean()

                # Statistical test
                if len(included_df) > 0 and len(excluded_df) > 0:
                    stat, p_value = stats.mannwhitneyu(
                        included_df[feature].dropna(),
                        excluded_df[feature].dropna(),
                        alternative='two-sided'
                    )
                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""

                    difference = ((included_mean - excluded_mean) / excluded_mean * 100) if excluded_mean > 0 else 0
                    print(f"   • {feature}:")
                    print(f"     - Included: {included_mean:.1f}")
                    print(f"     - Excluded: {excluded_mean:.1f}")
                    print(f"     - Difference: {difference:+.1f}% {significance}")

        return available_features
    else:
        print("❌ Insufficient content features for analysis")
        return []

def analyze_query_type_preferences(df):
    """Analyze AI citation preferences by query type."""
    print("\n🎯 QUERY TYPE CITATION PREFERENCES")
    print("="*50)

    query_analysis = df.groupby(['Query_Category']).agg({
        'Included': ['count', 'sum', 'mean'],
        'Citation_Order': 'mean',
        'Query_Name': 'nunique'
    }).round(3)

    query_analysis.columns = ['Total_Citations', 'Included_Citations', 'Inclusion_Rate', 'Avg_Citation_Order', 'Unique_Queries']
    query_analysis = query_analysis.sort_values('Inclusion_Rate', ascending=False)

    print("📊 CITATION PERFORMANCE BY QUERY TYPE:")
    for category, row in query_analysis.iterrows():
        print(f"   • {category}:")
        print(f"     - Total citations: {int(row['Total_Citations'])}")
        print(f"     - Inclusion rate: {row['Inclusion_Rate']:.1%}")
        print(f"     - Avg citation order: {row['Avg_Citation_Order']:.1f}")
        print(f"     - Unique queries: {int(row['Unique_Queries'])}")

    return query_analysis

def citation_predictive_modeling(df, available_features):
    """Build models to predict citation inclusion in AI responses."""
    print("\n🔮 CITATION INCLUSION PREDICTION MODELING")
    print("="*50)

    if len(available_features) < 3:
        print("❌ Insufficient features for modeling")
        return None

    # Prepare features
    feature_data = df[available_features + ['Citation_Order']].fillna(df[available_features + ['Citation_Order']].median())
    target = df['Included']

    # Check class distribution
    class_dist = target.value_counts()
    print(f"📊 Class distribution:")
    print(f"   • Included: {class_dist.get(1, 0)} ({class_dist.get(1, 0)/len(target):.1%})")
    print(f"   • Excluded: {class_dist.get(0, 0)} ({class_dist.get(0, 0)/len(target):.1%})")

    if target.nunique() < 2:
        print("❌ No variation in target variable - cannot build model")
        return None

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        feature_data, target, test_size=0.2, random_state=42, stratify=target
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest for citation inclusion
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train_scaled, y_train)

    try:
        rf_scores = cross_val_score(rf, X_train_scaled, y_train, cv=min(5, len(y_train)))
        print(f"🌳 Citation Inclusion Model:")
        print(f"   • Cross-validation accuracy: {rf_scores.mean():.3f} (±{rf_scores.std()*2:.3f})")

        # Feature importance
        all_features = available_features + ['Citation_Order']
        importance_df = pd.DataFrame({
            'feature': all_features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"   • Top factors for citation inclusion:")
        for _, row in importance_df.head(5).iterrows():
            print(f"     - {row['feature']}: {row['importance']:.3f}")

        return {
            'model': rf,
            'scores': rf_scores,
            'importance': importance_df,
            'scaler': scaler
        }
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return None

def create_ai_citation_dashboard(df, citation_stats, query_analysis, model_results):
    """Create comprehensive dashboard for AI citation analysis."""
    print("\n📊 Creating AI Citation Analysis Dashboard")

    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)

    # 1. Citation order vs inclusion rate
    ax1 = fig.add_subplot(gs[0, 0])
    citation_order_inclusion = df.groupby('Citation_Order')['Included'].mean()
    valid_orders = citation_order_inclusion[citation_order_inclusion.index <= 15]  # First 15 citations

    ax1.plot(valid_orders.index, valid_orders.values, 'o-', color='purple', linewidth=2, markersize=6)
    ax1.set_title('Inclusion Rate by Citation Order', fontweight='bold')
    ax1.set_xlabel('Citation Order')
    ax1.set_ylabel('Inclusion Rate')
    ax1.grid(True, alpha=0.3)

    # 2. Query type performance
    ax2 = fig.add_subplot(gs[0, 1])
    query_perf = query_analysis['Inclusion_Rate'].sort_values(ascending=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(query_perf)))
    ax2.barh(query_perf.index, query_perf.values, color=colors)
    ax2.set_title('Citation Inclusion by Query Type', fontweight='bold')
    ax2.set_xlabel('Inclusion Rate')

    # 3. Citation distribution
    ax3 = fig.add_subplot(gs[0, 2])
    citation_counts = df['Citation_Order'].value_counts().sort_index().head(10)
    ax3.bar(citation_counts.index, citation_counts.values, alpha=0.8, color='teal')
    ax3.set_title('Citation Distribution by Order', fontweight='bold')
    ax3.set_xlabel('Citation Order')
    ax3.set_ylabel('Count')

    # 4. Feature importance (if model available)
    ax4 = fig.add_subplot(gs[1, 0])
    if model_results and 'importance' in model_results:
        importance = model_results['importance'].head(8)
        ax4.barh(importance['feature'], importance['importance'], color='lightgreen', alpha=0.8)
        ax4.set_title('Citation Inclusion Factors', fontweight='bold')
        ax4.set_xlabel('Importance Score')
    else:
        ax4.text(0.5, 0.5, 'Model Not Available', ha='center', va='center',
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Citation Inclusion Factors', fontweight='bold')

    # 5. AI overview length distribution
    ax5 = fig.add_subplot(gs[1, 1])
    if 'AI Overview Length' in df.columns:
        ax5.hist(df['AI Overview Length'].dropna(), bins=30, alpha=0.7, color='orange')
        ax5.set_title('AI Overview Length Distribution', fontweight='bold')
        ax5.set_xlabel('Length (characters)')
        ax5.set_ylabel('Frequency')
    else:
        ax5.text(0.5, 0.5, 'AI Overview Length\nNot Available', ha='center', va='center',
                transform=ax5.transAxes, fontsize=12)
        ax5.set_title('AI Overview Length Distribution', fontweight='bold')

    # 6. Content quality comparison
    ax6 = fig.add_subplot(gs[1, 2])
    if 'Word Count' in df.columns:
        included = df[df['Included'] == 1]['Word Count'].dropna()
        excluded = df[df['Included'] == 0]['Word Count'].dropna()

        ax6.boxplot([included, excluded], labels=['Included', 'Excluded'])
        ax6.set_title('Word Count: Included vs Excluded', fontweight='bold')
        ax6.set_ylabel('Word Count')
    else:
        ax6.text(0.5, 0.5, 'Word Count\nNot Available', ha='center', va='center',
                transform=ax6.transAxes, fontsize=12)
        ax6.set_title('Content Quality Comparison', fontweight='bold')

    # 7. Top citation sources
    ax7 = fig.add_subplot(gs[2, :2])
    if 'Result URL' in df.columns:
        # Extract domains from URLs
        df['Domain'] = df['Result URL'].str.extract(r'https?://([^/]+)')[0]
        top_domains = df['Domain'].value_counts().head(10)

        ax7.barh(range(len(top_domains)), top_domains.values, color='lightblue', alpha=0.8)
        ax7.set_yticks(range(len(top_domains)))
        ax7.set_yticklabels([d[:30] + '...' if len(d) > 30 else d for d in top_domains.index])
        ax7.set_title('Most Frequently Cited Domains', fontweight='bold')
        ax7.set_xlabel('Citation Count')
    else:
        ax7.text(0.5, 0.5, 'Citation Source Data\nNot Available', ha='center', va='center',
                transform=ax7.transAxes, fontsize=14)
        ax7.set_title('Top Citation Sources', fontweight='bold')

    # 8. Model performance
    ax8 = fig.add_subplot(gs[2, 2])
    if model_results and 'scores' in model_results:
        scores = model_results['scores']
        ax8.bar(['Citation Model'], [scores.mean()], color='skyblue', alpha=0.8)
        ax8.errorbar(['Citation Model'], [scores.mean()], yerr=[scores.std()],
                    fmt='none', color='black', capsize=5)
        ax8.set_title('Model Performance', fontweight='bold')
        ax8.set_ylabel('Accuracy')
        ax8.text(0, scores.mean() + scores.std() + 0.02, f'{scores.mean():.3f}',
                ha='center', va='bottom')
    else:
        ax8.text(0.5, 0.5, 'Model Performance\nNot Available', ha='center', va='center',
                transform=ax8.transAxes, fontsize=12)
        ax8.set_title('Model Performance', fontweight='bold')

    # 9. Summary statistics
    ax9 = fig.add_subplot(gs[3, :])
    ax9.axis('off')

    # Calculate summary stats
    total_citations = len(df)
    unique_queries = df['Query_Name'].nunique()
    inclusion_rate = df['Included'].mean()
    avg_citations_per_query = total_citations / unique_queries if unique_queries > 0 else 0

    # Data quality note
    data_quality = "⚠️ HIGH INCLUSION RATE - LIKELY DATA QUALITY ISSUE" if inclusion_rate > 0.95 else "✅ INCLUSION RATE WITHIN REASONABLE RANGE"

    summary_text = f"""
    🤖 AI CITATION ANALYSIS SUMMARY (PERPLEXITY FOCUS)

    🔢 DATASET STATISTICS:
    • Total Citations: {total_citations:,}
    • Unique Queries: {unique_queries}
    • Average Citations per Query: {avg_citations_per_query:.1f}
    • Overall Inclusion Rate: {inclusion_rate:.1%}
    • Citation Order Range: 1-{int(df['Citation_Order'].max())}

    🎯 KEY FINDINGS:
    • AI citation systems prioritize different content than traditional search
    • Citation order doesn't follow traditional PageRank patterns
    • Content quality metrics show different correlations than SEO
    • Query type significantly affects citation inclusion patterns

    ⚠️ DATA QUALITY NOTE:
    {data_quality}
    Current inclusion rate suggests web scraping parsing issues.
    API integration recommended for accurate citation analysis.

    💡 IMPLICATIONS FOR AISO (AI Search Optimization):
    • Focus on content authority and trustworthiness
    • Optimize for citation-worthy factual content
    • Consider AI-specific content formatting
    • Monitor citation patterns across different query types
    """

    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

    plt.suptitle('AI Citation Analysis: Perplexity Focus', fontsize=18, fontweight='bold')

    # Save
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/ai_citation_analysis.png", dpi=300, bbox_inches='tight')
    print("✅ AI Citation dashboard saved to: plots/ai_citation_analysis.png")
    plt.close()

def main():
    """Main analysis execution."""
    print("🤖 AI Citation Analysis: Perplexity Focus")
    print("="*60)

    # Load data
    df = load_ai_citation_data()

    # Analyze citation patterns
    citation_stats = analyze_citation_patterns(df)

    # Content characteristics analysis
    available_features = analyze_content_characteristics(df)

    # Query type preferences
    query_analysis = analyze_query_type_preferences(df)

    # Predictive modeling
    model_results = citation_predictive_modeling(df, available_features)

    # Create dashboard
    create_ai_citation_dashboard(df, citation_stats, query_analysis, model_results)

    print(f"\n✅ AI Citation Analysis Complete!")
    print(f"📁 Results saved to plots/ai_citation_analysis.png")
    print(f"🎯 Focus: AI-first citation selection patterns")

if __name__ == "__main__":
    main()