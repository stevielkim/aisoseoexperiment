"""Content feature analysis for AI citations.

This module analyzes what content features predict AI search engine citations.
"""

import os
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from urllib.parse import urlparse

from ..analysis.statistical import test_correlations_with_fdr


def load_data(citation_file="data/processed/ai_serp_analysis.csv",
              source_file="data/processed/source_features.csv"):
    """
    Load citation data and source features.

    Parameters
    ----------
    citation_file : str, optional
        Path to citation data CSV
    source_file : str, optional
        Path to source features CSV

    Returns
    -------
    tuple
        (citations_df, source_features_df) or (None, None) if files not found

    Examples
    --------
    >>> citations_df, source_df = load_data()
    >>> print(f"Loaded {len(citations_df)} citations")
    """
    print("📊 Loading Data")
    print("=" * 60)

    # Load citation data
    if not os.path.exists(citation_file):
        print(f"❌ Citation data not found: {citation_file}")
        print("   Run parse_geo.py first to generate citation data.")
        return None, None

    citations_df = pd.read_csv(citation_file)
    print(f"✅ Loaded {len(citations_df)} citation records")

    # Load source features if available
    source_features_df = None
    if os.path.exists(source_file):
        source_features_df = pd.read_csv(source_file)
        print(f"✅ Loaded {len(source_features_df)} source feature records")
    else:
        print(f"⚠️  Source features not found: {source_file}")
        print("   Run fetch_source_features.py to extract content features.")

    return citations_df, source_features_df


def analyze_citation_patterns(df):
    """
    Analyze basic citation patterns across engines.

    Parameters
    ----------
    df : pd.DataFrame
        Citation DataFrame with Engine, Query, Result URL columns

    Returns
    -------
    tuple
        (engine_stats dict, valid_df) where engine_stats contains per-engine
        statistics

    Examples
    --------
    >>> stats, valid_df = analyze_citation_patterns(citations_df)
    >>> print(stats['Perplexity']['total_citations'])
    """
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
    """
    Analyze citation order patterns - which sources are cited first?

    Parameters
    ----------
    df : pd.DataFrame
        Citation DataFrame with Citation_Order column

    Returns
    -------
    pd.Series or None
        Citation order value counts, or None if no data available

    Examples
    --------
    >>> order_counts = analyze_citation_order(citations_df)
    >>> print(f"Position 1: {order_counts[1]} citations")
    """
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


def extract_domain(url):
    """
    Extract domain from URL.

    Parameters
    ----------
    url : str
        Full URL

    Returns
    -------
    str
        Domain without www prefix, or empty string if invalid

    Examples
    --------
    >>> extract_domain("https://www.example.com/page")
    'example.com'
    """
    if pd.isna(url) or url == '':
        return ''
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except:
        return ''


def classify_domain_type(domain):
    """
    Classify domain by TLD.

    Parameters
    ----------
    domain : str
        Domain name

    Returns
    -------
    str
        Domain type: 'Educational', 'Government', 'Organization', or 'Commercial'

    Examples
    --------
    >>> classify_domain_type("harvard.edu")
    'Educational'
    """
    if domain.endswith('.edu'):
        return 'Educational'
    elif domain.endswith('.gov'):
        return 'Government'
    elif domain.endswith('.org'):
        return 'Organization'
    else:
        return 'Commercial'


def analyze_domain_patterns(df):
    """
    Analyze which domains get cited most frequently.

    Parameters
    ----------
    df : pd.DataFrame
        Citation DataFrame with Result URL column

    Returns
    -------
    tuple
        (domain_counts, type_counts) - Series of citation counts by domain and type

    Examples
    --------
    >>> domain_counts, type_counts = analyze_domain_patterns(citations_df)
    >>> print(f"Top domain: {domain_counts.index[0]} with {domain_counts.values[0]} citations")
    """
    print("\n🌐 DOMAIN ANALYSIS")
    print("=" * 60)

    # Extract domains from URLs
    df['Domain'] = df['Result URL'].apply(extract_domain)

    # Top cited domains
    domain_counts = df[df['Domain'] != '']['Domain'].value_counts()

    print("\n🏆 Most Frequently Cited Domains:")
    for i, (domain, count) in enumerate(domain_counts.head(20).items(), 1):
        print(f"   {i:2}. {domain}: {count} citations")

    # Domain type analysis
    df['Domain_Type'] = df['Domain'].apply(classify_domain_type)

    print("\n📊 Citations by Domain Type:")
    type_counts = df[df['Domain'] != '']['Domain_Type'].value_counts()
    for domain_type, count in type_counts.items():
        pct = count / type_counts.sum() * 100
        print(f"   • {domain_type}: {count} ({pct:.1f}%)")

    return domain_counts, type_counts


def normalize_url(url):
    """
    Normalize URL for matching.

    Parameters
    ----------
    url : str
        Full URL

    Returns
    -------
    str
        Normalized URL (lowercase, no protocol, no trailing slash, no www)

    Examples
    --------
    >>> normalize_url("https://www.Example.com/page/")
    'example.com/page'
    """
    if pd.isna(url) or url == '':
        return ''
    url = str(url).lower()
    url = url.replace('https://', '').replace('http://', '')
    url = url.rstrip('/')
    if url.startswith('www.'):
        url = url[4:]
    return url


def analyze_source_content_features(citations_df, source_df):
    """
    Analyze content features of cited sources.

    Parameters
    ----------
    citations_df : pd.DataFrame
        Citation data
    source_df : pd.DataFrame
        Source features data

    Returns
    -------
    tuple or None
        (merged_df, feature_stats, available_features) or None if no data

    Examples
    --------
    >>> merged_df, stats, features = analyze_source_content_features(citations_df, source_df)
    >>> print(f"Average word count: {stats['word_count']['mean']:.0f}")
    """
    print("\n📝 SOURCE CONTENT FEATURE ANALYSIS")
    print("=" * 60)

    if source_df is None or len(source_df) == 0:
        print("❌ No source features available")
        print("   Run fetch_source_features.py to extract content features")
        return None

    # Normalize URLs for matching
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
    """
    Analyze relationship between citation order and content features.

    Uses automatic Pearson/Spearman selection with FDR correction.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Merged citation + feature data
    available_features : list of str
        Feature column names to analyze

    Returns
    -------
    tuple or None
        (correlations dict, correlation_results list) or None if insufficient data

    Examples
    --------
    >>> correlations, results = analyze_citation_order_vs_features(merged_df, features)
    >>> for r in results:
    ...     if r['significant']:
    ...         print(f"{r['feature']}: {r['coefficient']:.3f}")
    """
    print("\n🔗 CITATION ORDER vs CONTENT FEATURES")
    print("=" * 60)

    if merged_df is None or len(merged_df) == 0:
        return None

    # Filter to rows with citation order
    ordered_df = merged_df[merged_df['Citation_Order'].notna()].copy()

    if len(ordered_df) < 10:
        print("❌ Not enough data for citation order analysis")
        return None

    # Correlation with FDR correction
    print("\n📈 Correlation with Citation Order (lower = cited earlier):")
    print("   (Negative correlation = feature associated with being cited earlier)")
    print("-" * 50)

    # Test correlations with FDR correction
    correlation_results = test_correlations_with_fdr(
        ordered_df,
        'Citation_Order',
        available_features,
        alpha=0.05
    )

    if not correlation_results:
        print("   No features available for correlation analysis")
        return None

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
    """
    Build a model to identify which features predict early citation.

    Uses Random Forest with train-test split and proper evaluation.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Merged citation + feature data
    available_features : list of str
        Feature column names to use in model

    Returns
    -------
    tuple or None
        (importance_df, cv_scores) or None if insufficient data

    Examples
    --------
    >>> importance_df, cv_scores = build_feature_importance_model(merged_df, features)
    >>> print(importance_df.head())
    """
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

    # Train-test split for proper evaluation
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
