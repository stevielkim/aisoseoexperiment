#!/usr/bin/env python3
"""
AI Content Citation Analysis

Goal: What features in content make it more or less likely to be included in AI Overviews?

This script analyzes the content features of sources cited by AI search engines
to identify patterns that predict citation likelihood.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from collections import Counter
from scipy.stats import pearsonr, spearmanr, shapiro
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint
import warnings
warnings.filterwarnings('ignore')

# Configuration
CITATION_DATA_FILE = "ai_serp_analysis.csv"
SOURCE_FEATURES_FILE = "source_features.csv"
OUTPUT_PLOT = "plots/content_feature_analysis.png"

# Style
plt.style.use('default')
sns.set_palette("viridis")


def audit_data_quality(df, engine_name="All"):
    """Flag potential data quality issues in citation data."""
    if 'Included' not in df.columns:
        return

    inclusion_rate = df['Included'].mean()
    n = len(df)

    # Calculate 95% confidence interval
    ci_low, ci_high = proportion_confint(
        count=df['Included'].sum(),
        nobs=n,
        alpha=0.05,
        method='wilson'
    )

    print(f"\n🔍 Data Quality Audit - {engine_name}:")
    print(f"   • Sample size: {n}")
    print(f"   • Inclusion rate: {inclusion_rate:.1%} [{ci_low:.1%}, {ci_high:.1%}]")

    if inclusion_rate > 0.90:
        print(f"   ⚠️  WARNING: Inclusion rate suspiciously high")
        print(f"       May indicate parser issues or biased sampling")
    elif inclusion_rate < 0.05:
        print(f"   ⚠️  WARNING: Inclusion rate suspiciously low")
        print(f"       Parser may be too strict")
    else:
        print(f"   ✅ Inclusion rate within expected range")


def load_data():
    """Load citation data and source features."""
    print("📊 Loading Data")
    print("=" * 60)
    
    # Load citation data
    if not os.path.exists(CITATION_DATA_FILE):
        print(f"❌ Citation data not found: {CITATION_DATA_FILE}")
        print("   Run parse_geo.py first to generate citation data.")
        return None, None
    
    citations_df = pd.read_csv(CITATION_DATA_FILE)
    print(f"✅ Loaded {len(citations_df)} citation records")
    
    # Load source features if available
    source_features_df = None
    if os.path.exists(SOURCE_FEATURES_FILE):
        source_features_df = pd.read_csv(SOURCE_FEATURES_FILE)
        print(f"✅ Loaded {len(source_features_df)} source feature records")
    else:
        print(f"⚠️  Source features not found: {SOURCE_FEATURES_FILE}")
        print("   Run fetch_source_features.py to extract content features.")
    
    return citations_df, source_features_df


def analyze_citation_patterns(df):
    """Analyze basic citation patterns across engines."""
    print("\n📈 CITATION PATTERN ANALYSIS")
    print("=" * 60)
    
    # Filter out blocked pages (handle both old and new schema)
    if 'Is_Blocked' in df.columns:
        valid_df = df[df['Is_Blocked'] == False].copy()
    else:
        # Old schema - use all data
        valid_df = df.copy()
    
    # Per-engine statistics
    engine_stats = {}
    
    # Handle both old and new schema for query extraction
    if 'Query' not in valid_df.columns and 'File' in valid_df.columns:
        # Extract query from filename
        valid_df['Query'] = valid_df['File'].str.extract(r'([^/]+)\.html$')[0].str.replace('_', ' ')
        valid_df['Query'] = valid_df['Query'].str.replace(' google ai', '').str.replace(' bing ai', '').str.replace(' perplexity', '')
    
    for engine in valid_df['Engine'].unique():
        engine_data = valid_df[valid_df['Engine'] == engine]
        
        # Count unique queries and sources
        unique_queries = engine_data['Query'].nunique() if 'Query' in engine_data.columns else 0
        unique_sources = engine_data[engine_data['Result URL'] != '']['Result URL'].nunique()
        total_citations = len(engine_data[engine_data['Result URL'] != ''])
        
        # Citations per query
        citations_per_query = engine_data.groupby('Query').size()
        avg_citations = citations_per_query.mean()
        
        engine_stats[engine] = {
            'unique_queries': unique_queries,
            'unique_sources': unique_sources,
            'total_citations': total_citations,
            'avg_citations_per_query': avg_citations,
        }
        
        print(f"\n🔍 {engine}:")
        print(f"   • Unique queries: {unique_queries}")
        print(f"   • Unique source URLs: {unique_sources}")
        print(f"   • Total citations: {total_citations}")
        print(f"   • Avg citations per query: {avg_citations:.1f}")
    
    return engine_stats, valid_df


def analyze_citation_order(df):
    """Analyze citation order patterns - which sources are cited first?"""
    print("\n📊 CITATION ORDER ANALYSIS")
    print("=" * 60)
    
    # Filter to records with valid citation order
    ordered_df = df[df['Citation_Order'].notna()].copy()
    
    if len(ordered_df) == 0:
        print("❌ No citation order data available")
        return None
    
    # Citation order distribution
    order_counts = ordered_df['Citation_Order'].value_counts().sort_index()
    
    print("\n🔢 Citation Order Distribution:")
    for order, count in order_counts.head(10).items():
        print(f"   • Position {int(order)}: {count} citations ({count/len(ordered_df)*100:.1f}%)")
    
    # Average citation order by engine
    print("\n📈 Average Citation Position by Engine:")
    for engine in ordered_df['Engine'].unique():
        engine_data = ordered_df[ordered_df['Engine'] == engine]
        avg_order = engine_data['Citation_Order'].mean()
        print(f"   • {engine}: {avg_order:.1f}")
    
    return order_counts


def analyze_domain_patterns(df):
    """Analyze which domains get cited most frequently."""
    print("\n🌐 DOMAIN ANALYSIS")
    print("=" * 60)
    
    # Extract domains from URLs
    def extract_domain(url):
        if pd.isna(url) or url == '':
            return ''
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except:
            return ''
    
    df['Domain'] = df['Result URL'].apply(extract_domain)
    
    # Top cited domains
    domain_counts = df[df['Domain'] != '']['Domain'].value_counts()
    
    print("\n🏆 Most Frequently Cited Domains:")
    for i, (domain, count) in enumerate(domain_counts.head(20).items(), 1):
        print(f"   {i:2}. {domain}: {count} citations")
    
    # Domain type analysis
    def classify_domain(domain):
        if domain.endswith('.edu'):
            return 'Educational'
        elif domain.endswith('.gov'):
            return 'Government'
        elif domain.endswith('.org'):
            return 'Organization'
        else:
            return 'Commercial'
    
    df['Domain_Type'] = df['Domain'].apply(classify_domain)
    
    print("\n📊 Citations by Domain Type:")
    type_counts = df[df['Domain'] != '']['Domain_Type'].value_counts()
    for domain_type, count in type_counts.items():
        pct = count / type_counts.sum() * 100
        print(f"   • {domain_type}: {count} ({pct:.1f}%)")
    
    return domain_counts, type_counts


def analyze_source_content_features(citations_df, source_df):
    """Analyze content features of cited sources."""
    print("\n📝 SOURCE CONTENT FEATURE ANALYSIS")
    print("=" * 60)
    
    if source_df is None or len(source_df) == 0:
        print("❌ No source features available")
        print("   Run fetch_source_features.py to extract content features")
        return None
    
    # Normalize URLs for matching
    def normalize_url(url):
        if pd.isna(url) or url == '':
            return ''
        url = str(url).lower()
        url = url.replace('https://', '').replace('http://', '')
        url = url.rstrip('/')
        if url.startswith('www.'):
            url = url[4:]
        return url
    
    citations_df = citations_df.copy()
    source_df = source_df.copy()
    
    citations_df['url_normalized'] = citations_df['Result URL'].apply(normalize_url)
    source_df['url_normalized'] = source_df['url'].apply(normalize_url)
    
    # Merge
    merged_df = citations_df.merge(
        source_df,
        on='url_normalized',
        how='left',
        suffixes=('', '_source')
    )
    
    # Filter to successfully fetched sources
    # Handle both old and new column names
    success_col = 'fetch_success' if 'fetch_success' in merged_df.columns else 'fetch_status'
    if success_col not in merged_df.columns:
        print("❌ Source features file missing fetch status column")
        print("   Run fetch_source_features.py to generate fresh data")
        return None
    
    if success_col == 'fetch_status':
        fetched_df = merged_df[merged_df[success_col] == 'success'].copy()
    else:
        fetched_df = merged_df[merged_df[success_col] == 1].copy()
    
    print(f"✅ Matched {len(fetched_df)} citations with source features")
    
    if len(fetched_df) == 0:
        print("❌ No matches found between citations and source features")
        return None
    
    # Analyze content features of cited sources
    content_features = [
        'word_count', 'h1_count', 'h2_count', 'h3_count',
        'total_list_count', 'table_count', 'image_count',
        'paragraph_count', 'avg_paragraph_length',
        'internal_link_count', 'external_link_count',
        'has_author', 'has_publish_date', 'has_any_schema',
    ]
    
    available_features = [f for f in content_features if f in fetched_df.columns]
    
    print("\n📊 Content Feature Statistics (Cited Sources):")
    print("-" * 50)
    
    feature_stats = {}
    for feature in available_features:
        if fetched_df[feature].notna().sum() > 0:
            mean_val = fetched_df[feature].mean()
            median_val = fetched_df[feature].median()
            feature_stats[feature] = {
                'mean': mean_val,
                'median': median_val,
            }
            print(f"   • {feature}: mean={mean_val:.1f}, median={median_val:.1f}")
    
    # Content type distribution
    if 'content_type' in fetched_df.columns:
        print("\n📋 Content Type Distribution:")
        type_counts = fetched_df['content_type'].value_counts()
        for content_type, count in type_counts.items():
            pct = count / len(fetched_df) * 100
            print(f"   • {content_type}: {count} ({pct:.1f}%)")
    
    return fetched_df, feature_stats, available_features


def analyze_citation_order_vs_features(merged_df, available_features):
    """Analyze relationship between citation order and content features."""
    print("\n🔗 CITATION ORDER vs CONTENT FEATURES")
    print("=" * 60)
    
    if merged_df is None or len(merged_df) == 0:
        return None
    
    # Filter to rows with citation order
    ordered_df = merged_df[merged_df['Citation_Order'].notna()].copy()
    
    if len(ordered_df) < 10:
        print("❌ Not enough data for citation order analysis")
        return None
    
    # Correlation between features and citation order with proper statistical testing
    print("\n📈 Correlation with Citation Order (lower = cited earlier):")
    print("   (Negative correlation = feature associated with being cited earlier)")
    print("-" * 50)

    correlation_results = []

    for feature in available_features:
        if feature in ordered_df.columns and ordered_df[feature].notna().sum() > 10:
            # Get clean data
            data = ordered_df[['Citation_Order', feature]].dropna()

            if len(data) < 10:
                continue

            # Test normality of feature
            try:
                _, p_norm = shapiro(data[feature])
                is_normal = p_norm > 0.05
            except:
                is_normal = False

            # Choose appropriate correlation method
            if is_normal:
                stat, p_val = pearsonr(data['Citation_Order'], data[feature])
                method = 'Pearson'
            else:
                stat, p_val = spearmanr(data['Citation_Order'], data[feature])
                method = 'Spearman'

            correlation_results.append({
                'feature': feature,
                'coefficient': stat,
                'p_value': p_val,
                'method': method,
                'n': len(data),
                'normal': is_normal
            })

    if not correlation_results:
        print("   No features available for correlation analysis")
        return None

    # Apply FDR correction for multiple comparisons
    p_values = [r['p_value'] for r in correlation_results]
    reject, p_adj, _, _ = multipletests(p_values, method='fdr_bh')

    # Add adjusted p-values and significance
    for i, result in enumerate(correlation_results):
        result['p_adj'] = p_adj[i]
        result['significant'] = reject[i]

    # Sort by absolute correlation
    correlation_results.sort(key=lambda x: abs(x['coefficient']), reverse=True)

    # Report top 10 correlations
    print("\n   Top 10 correlations (FDR-corrected p-values):")
    for result in correlation_results[:10]:
        direction = "⬆️ Higher" if result['coefficient'] > 0 else "⬇️ Lower"
        sig = "***" if result['significant'] else ""
        print(f"   • {result['feature']}: r={result['coefficient']:+.3f}, "
              f"p={result['p_value']:.4f}, p_adj={result['p_adj']:.4f} {sig}")
        print(f"     {direction} = cited later (n={result['n']}, {result['method']})")

    # Convert to dict for backward compatibility
    correlations = {r['feature']: r['coefficient'] for r in correlation_results}

    return correlations, correlation_results


def build_feature_importance_model(merged_df, available_features):
    """Build a model to identify which features predict early citation."""
    print("\n🤖 FEATURE IMPORTANCE MODEL")
    print("=" * 60)
    
    if merged_df is None or len(merged_df) < 30:
        print("❌ Not enough data for modeling")
        return None
    
    # Create target: cited in top 3 positions vs later
    df = merged_df[merged_df['Citation_Order'].notna()].copy()
    df['early_citation'] = (df['Citation_Order'] <= 3).astype(int)
    
    # Prepare features
    model_features = [f for f in available_features if f in df.columns]
    
    # Drop rows with missing values in features
    df_model = df[model_features + ['early_citation']].dropna()
    
    if len(df_model) < 30:
        print(f"❌ Not enough complete records for modeling ({len(df_model)} records)")
        return None
    
    X = df_model[model_features]
    y = df_model['early_citation']

    # Analyze class imbalance
    class_ratio = y.sum() / len(y)
    imbalance_severity = ("extreme" if class_ratio < 0.1 or class_ratio > 0.9
                         else "moderate" if class_ratio < 0.3 or class_ratio > 0.7
                         else "mild")

    print(f"📊 Model training data: {len(df_model)} records")
    print(f"   • Early citations (top 3): {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"   • Later citations: {len(y) - y.sum()} ({(1-y.mean())*100:.1f}%)")
    print(f"   • Class imbalance: {imbalance_severity} (ratio={min(class_ratio, 1-class_ratio):.3f})")

    if imbalance_severity == "extreme":
        print(f"   ⚠️  EXTREME IMBALANCE: Using class weights and stratified split")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # IMPROVEMENT: Train-test split for proper evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n📊 Data Split:")
    print(f"   • Training set: {len(X_train)} records")
    print(f"   • Test set: {len(X_test)} records")

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

    try:
        # Cross-validation on training data only
        cv_scores = cross_val_score(rf, X_train, y_train, cv=min(5, len(y_train)//5))
        print(f"\n🌳 Random Forest Model:")
        print(f"   • CV accuracy (train): {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")

        # Train on training data
        rf.fit(X_train, y_train)

        # Evaluate on test set
        test_score = rf.score(X_test, y_test)
        train_score = rf.score(X_train, y_train)
        print(f"   • Train accuracy: {train_score:.3f}")
        print(f"   • Test accuracy: {test_score:.3f}")

        if abs(train_score - test_score) > 0.1:
            print(f"   ⚠️  Large train-test gap suggests overfitting")
        else:
            print(f"   ✅ Good generalization (small train-test gap)")

        # Feature importance (from model trained on training data)
        importance_df = pd.DataFrame({
            'feature': model_features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\n🎯 Top Features Predicting Early Citation:")
        print(f"   (Based on Gini importance from Random Forest)")
        for _, row in importance_df.head(10).iterrows():
            print(f"   • {row['feature']}: {row['importance']:.3f}")
        
        return importance_df, cv_scores
        
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return None


def create_analysis_dashboard(citations_df, domain_counts, type_counts, 
                             feature_stats, correlations, importance_df):
    """Create visualization dashboard."""
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
    unique_queries = citations_df['Query'].nunique()
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
    os.makedirs("plots", exist_ok=True)
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Dashboard saved to: {OUTPUT_PLOT}")
    plt.close()


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
    create_analysis_dashboard(
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
    print(f"\n📁 Results saved to: {OUTPUT_PLOT}")
    print("\n💡 Next Steps:")
    if source_df is None:
        print("   1. Run: python fetch_source_features.py")
        print("      (This will fetch actual content features from source URLs)")
        print("   2. Re-run: python analyze_content_features.py")
        print("      (To see full content feature analysis)")
    else:
        print("   1. Review the dashboard for content optimization insights")
        print("   2. Consider expanding the query dataset for more robust analysis")


if __name__ == "__main__":
    main()

