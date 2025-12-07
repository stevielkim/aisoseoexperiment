"""Traditional SEO analysis for Google AI + Bing AI.

This module analyzes whether traditional SEO factors predict inclusion
in AI Overviews for traditional search engines.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency

from ..analysis.statistical import test_correlations_with_fdr, cramers_v


def load_traditional_seo_data(data_file="data/processed/ai_serp_analysis.csv"):
    """
    Load and prepare data for traditional SEO analysis (Google AI + Bing AI only).

    Parameters
    ----------
    data_file : str, optional
        Path to citation data CSV

    Returns
    -------
    pd.DataFrame
        Filtered and prepared DataFrame

    Examples
    --------
    >>> df = load_traditional_seo_data()
    >>> print(df['Engine'].unique())
    ['Google AI' 'Bing AI']
    """
    print("📊 Loading Traditional SEO Data (Google AI + Bing AI)")
    print("="*60)

    df = pd.read_csv(data_file)

    # Filter to only traditional search engines
    traditional_engines = ['Google AI', 'Bing AI']
    df = df[df['Engine'].isin(traditional_engines)].copy()

    # Clean data
    df['Included'] = df['Included'].astype(int)
    df['Query_Name'] = df['File'].str.extract(r'([^/]+)\.html$')[0].str.replace('_', ' ')

    # Categorize queries
    df['Query_Category'] = df['Query_Name'].apply(categorize_query)

    print(f"✅ Filtered to traditional search engines:")
    print(f"   • Total records: {len(df):,}")
    print(f"   • Google AI: {len(df[df['Engine'] == 'Google AI']):,}")
    print(f"   • Bing AI: {len(df[df['Engine'] == 'Bing AI']):,}")
    print(f"   • Unique queries: {df['Query_Name'].nunique()}")
    print(f"   • Overall inclusion rate: {df['Included'].mean():.1%}")

    return df


def categorize_query(query):
    """
    Categorize query by type.

    Parameters
    ----------
    query : str
        Query text

    Returns
    -------
    str
        Query category: 'How-to', 'Informational', 'Comparison', 'Best-of', or 'Other'

    Examples
    --------
    >>> categorize_query("how to bake bread")
    'How-to'
    >>> categorize_query("python vs java")
    'Comparison'
    """
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


def analyze_traditional_ranking_factors(df):
    """
    Analyze traditional SEO factors in AI Overview inclusion.

    Uses automatic Pearson/Spearman selection with FDR correction.

    Parameters
    ----------
    df : pd.DataFrame
        Traditional SEO data with features and Included column

    Returns
    -------
    tuple
        (correlations Series, available_features list, correlation_results list)
        or (None, available_features, None) if insufficient data

    Examples
    --------
    >>> corr, features, results = analyze_traditional_ranking_factors(df)
    >>> print(corr['Word Count'])
    0.234
    """
    print("\n🔍 TRADITIONAL SEO FACTOR ANALYSIS")
    print("="*50)

    # Define traditional SEO features
    seo_features = [
        'Word Count', 'H1 Count', 'H2 Count', 'H3 Count', 'MetaDesc Length',
        'List Count', 'Image Count', 'Page Rank'
    ]

    # Filter to available features
    available_features = [f for f in seo_features if f in df.columns]
    print(f"📋 Available SEO features: {', '.join(available_features)}")

    # Correlation analysis with FDR correction
    if len(available_features) >= 3:
        feature_data = df[available_features + ['Included']].fillna(0)

        # Use shared statistical utility
        correlation_results = test_correlations_with_fdr(
            feature_data,
            'Included',
            available_features,
            alpha=0.05
        )

        if not correlation_results:
            print("❌ No valid correlations found")
            return None, available_features, None

        # Sort by absolute correlation
        correlation_results.sort(key=lambda x: abs(x['coefficient']), reverse=True)

        print(f"\n🔗 SEO FEATURE CORRELATIONS WITH AI OVERVIEW INCLUSION:")
        print(f"   (Using Pearson or Spearman based on normality tests)")
        print(f"   (P-values adjusted for multiple comparisons using FDR correction)")

        for result in correlation_results:
            sig_marker = "***" if result['significant'] else ""
            print(f"   • {result['feature']}: {result['coefficient']:+.3f} "
                  f"(p_adj={result['p_adj']:.4f}) {sig_marker}")

        # Create simple correlations dict for backward compatibility
        correlations = pd.Series({r['feature']: r['coefficient'] for r in correlation_results})

        return correlations, available_features, correlation_results
    else:
        print("❌ Insufficient features for correlation analysis")
        return None, available_features, None


def analyze_by_engine(df):
    """
    Compare traditional SEO performance between Google AI and Bing AI.

    Parameters
    ----------
    df : pd.DataFrame
        Traditional SEO data

    Returns
    -------
    dict
        Engine-specific analysis results

    Examples
    --------
    >>> engine_analysis = analyze_by_engine(df)
    >>> print(engine_analysis['Google AI']['inclusion_rate'])
    0.87
    """
    print("\n⚖️ ENGINE COMPARISON: GOOGLE AI vs BING AI")
    print("="*50)

    engine_analysis = {}

    for engine in ['Google AI', 'Bing AI']:
        engine_data = df[df['Engine'] == engine]

        print(f"\n🔧 {engine}:")
        print(f"   • Total results: {len(engine_data):,}")
        print(f"   • Inclusion rate: {engine_data['Included'].mean():.1%}")
        print(f"   • Avg page rank: {engine_data['Page Rank'].mean():.1f}")

        # Rank-based analysis
        rank_analysis = engine_data.groupby('Page Rank')['Included'].agg(['count', 'sum', 'mean']).head(10)
        print(f"   • Rank 1-3 inclusion: {engine_data[engine_data['Page Rank'] <= 3]['Included'].mean():.1%}")
        print(f"   • Rank 4-10 inclusion: {engine_data[(engine_data['Page Rank'] >= 4) & (engine_data['Page Rank'] <= 10)]['Included'].mean():.1%}")

        engine_analysis[engine] = {
            'data': engine_data,
            'inclusion_rate': engine_data['Included'].mean(),
            'rank_analysis': rank_analysis
        }

    # Statistical comparison with assumption validation and effect size
    google_included = df[df['Engine'] == 'Google AI']['Included']
    bing_included = df[df['Engine'] == 'Bing AI']['Included']

    # Chi-square test with assumption validation
    contingency = pd.crosstab(df['Engine'], df['Included'])
    chi2, p_value, dof, expected = chi2_contingency(contingency)

    # Validate chi-square assumptions (expected frequencies >= 5)
    min_expected = expected.min()

    # Calculate Cramér's V effect size
    n = contingency.sum().sum()
    r, c = contingency.shape
    cramers_v_value = cramers_v(chi2, n, r, c)

    print(f"\n📊 STATISTICAL SIGNIFICANCE:")
    print(f"   • Chi-square test: χ² = {chi2:.3f}, p = {p_value:.4f}")

    if min_expected < 5:
        print(f"   ⚠️  WARNING: Chi-square assumption violated (min expected = {min_expected:.1f})")
        print(f"       Results may be unreliable (expected frequencies should be ≥ 5)")
    else:
        print(f"   ✅ Chi-square assumptions met (min expected = {min_expected:.1f})")

    print(f"   • Cramér's V (effect size): {cramers_v_value:.3f}", end="")
    if cramers_v_value < 0.1:
        print(" (negligible)")
    elif cramers_v_value < 0.3:
        print(" (small)")
    elif cramers_v_value < 0.5:
        print(" (medium)")
    else:
        print(" (large)")

    print(f"   • Significant difference: {'Yes' if p_value < 0.05 else 'No'}")

    return engine_analysis


def predictive_modeling(df, available_features):
    """
    Build predictive models for AI Overview inclusion.

    Trains Random Forest and Logistic Regression with train-test split.

    Parameters
    ----------
    df : pd.DataFrame
        Traditional SEO data
    available_features : list of str
        Feature column names to use in models

    Returns
    -------
    dict or None
        Model results or None if insufficient data

    Examples
    --------
    >>> models = predictive_modeling(df, ['Word Count', 'H1 Count'])
    >>> print(models['random_forest']['test_score'])
    0.812
    """
    print("\n🤖 PREDICTIVE MODELING: AI OVERVIEW INCLUSION")
    print("="*50)

    if len(available_features) < 3:
        print("❌ Insufficient features for modeling")
        return None

    # Prepare data
    feature_data = df[available_features].fillna(df[available_features].median())
    target = df['Included']

    # Analyze class imbalance
    class_ratio = target.mean()
    imbalance_severity = ("extreme" if class_ratio < 0.1 or class_ratio > 0.9
                         else "moderate" if class_ratio < 0.3 or class_ratio > 0.7
                         else "mild")

    print(f"📊 Model training data: {len(df)} records")
    print(f"   • Included in AI Overview: {target.sum()} ({class_ratio*100:.1f}%)")
    print(f"   • Not included: {len(target) - target.sum()} ({(1-class_ratio)*100:.1f}%)")
    print(f"   • Class imbalance: {imbalance_severity} (ratio={min(class_ratio, 1-class_ratio):.3f})")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        feature_data, target, test_size=0.2, random_state=42, stratify=target
    )

    print(f"\n📊 Data Split:")
    print(f"   • Training set: {len(X_train)} records")
    print(f"   • Test set: {len(X_test)} records")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {}

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5)
    rf.fit(X_train_scaled, y_train)

    # Evaluate on test set
    train_score = rf.score(X_train_scaled, y_train)
    test_score = rf.score(X_test_scaled, y_test)

    print(f"\n🌳 Random Forest:")
    print(f"   • CV accuracy (train): {rf_scores.mean():.3f} (±{rf_scores.std():.3f})")
    print(f"   • Train accuracy: {train_score:.3f}")
    print(f"   • Test accuracy: {test_score:.3f}")

    if abs(train_score - test_score) > 0.1:
        print(f"   ⚠️  Large train-test gap suggests overfitting")
    else:
        print(f"   ✅ Good generalization (small train-test gap)")

    # Feature importance
    importance_df = pd.DataFrame({
        'feature': available_features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"   • Top 3 important features:")
    for _, row in importance_df.head(3).iterrows():
        print(f"     - {row['feature']}: {row['importance']:.3f}")

    models['random_forest'] = {
        'model': rf,
        'scores': rf_scores,
        'train_score': train_score,
        'test_score': test_score,
        'importance': importance_df
    }

    # Logistic Regression with coefficient interpretation
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr_scores = cross_val_score(lr, X_train_scaled, y_train, cv=5)
    lr.fit(X_train_scaled, y_train)

    # Evaluate on test set
    lr_train_score = lr.score(X_train_scaled, y_train)
    lr_test_score = lr.score(X_test_scaled, y_test)

    print(f"\n📈 Logistic Regression:")
    print(f"   • CV accuracy (train): {lr_scores.mean():.3f} (±{lr_scores.std():.3f})")
    print(f"   • Train accuracy: {lr_train_score:.3f}")
    print(f"   • Test accuracy: {lr_test_score:.3f}")

    if abs(lr_train_score - lr_test_score) > 0.1:
        print(f"   ⚠️  Large train-test gap suggests overfitting")
    else:
        print(f"   ✅ Good generalization (small train-test gap)")

    # Interpret coefficients as odds ratios
    coefficients = pd.DataFrame({
        'feature': available_features,
        'coefficient': lr.coef_[0],
        'odds_ratio': np.exp(lr.coef_[0])
    }).sort_values('coefficient', ascending=False)

    print(f"\n   💡 Logistic Regression Interpretation (Odds Ratios):")
    for _, row in coefficients.head(3).iterrows():
        direction = "increases" if row['odds_ratio'] > 1 else "decreases"
        pct_change = abs((row['odds_ratio'] - 1) * 100)
        print(f"     • {row['feature']}: OR={row['odds_ratio']:.2f}")
        print(f"       → 1-unit increase {direction} odds by {pct_change:.1f}%")

    models['logistic_regression'] = {
        'model': lr,
        'scores': lr_scores,
        'train_score': lr_train_score,
        'test_score': lr_test_score,
        'coefficients': coefficients
    }

    return models
