"""Statistical utilities for SEO vs AISO analysis.

This module provides statistical functions with proper multiple comparison
correction, confidence intervals, and effect size calculations.
"""

import numpy as np
import pandas as pd
from scipy.stats import shapiro, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint


def audit_data_quality(df, dataset_name="Dataset"):
    """
    Flag potential data quality issues in citation data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing citation data with 'Included' column
    dataset_name : str, optional
        Name of dataset for reporting, by default "Dataset"

    Returns
    -------
    None
        Prints quality audit results to console

    Examples
    --------
    >>> df = pd.read_csv('citations.csv')
    >>> audit_data_quality(df, "Perplexity")
    🔍 Data Quality Audit - Perplexity:
       • Sample size: 190
       • Inclusion rate: 95.3% [91.2%, 97.8%]
       ⚠️  WARNING: Inclusion rate suspiciously high
    """
    if 'Included' not in df.columns:
        print(f"\n⚠️  Data Quality Audit - {dataset_name}:")
        print(f"   'Included' column not found - cannot assess inclusion rate")
        return

    inclusion_rate = df['Included'].mean()
    n = len(df)

    # Calculate 95% confidence interval using Wilson score
    ci_low, ci_high = proportion_confint(
        count=df['Included'].sum(),
        nobs=n,
        alpha=0.05,
        method='wilson'
    )

    print(f"\n🔍 Data Quality Audit - {dataset_name}:")
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


def test_correlations_with_fdr(data, target_col, feature_cols, alpha=0.05):
    """
    Test correlations between target and features with FDR correction.

    Automatically chooses Pearson or Spearman correlation based on normality
    test, and applies Benjamini-Hochberg FDR correction for multiple comparisons.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing target and feature columns
    target_col : str
        Name of target column
    feature_cols : list of str
        Names of feature columns to test
    alpha : float, optional
        Significance level, by default 0.05

    Returns
    -------
    list of dict
        List of correlation results with keys:
        - feature: feature name
        - coefficient: correlation coefficient
        - p_value: raw p-value
        - p_adj: FDR-adjusted p-value
        - significant: bool, whether significant after FDR correction
        - method: 'Pearson' or 'Spearman'
        - n: sample size
        - normal: bool, whether feature is normally distributed

    Examples
    --------
    >>> results = test_correlations_with_fdr(
    ...     df, 'citation_order', ['word_count', 'h1_count', 'h2_count']
    ... )
    >>> for r in results:
    ...     if r['significant']:
    ...         print(f"{r['feature']}: r={r['coefficient']:.3f}, p={r['p_adj']:.4f}")
    """
    correlation_results = []

    for feature in feature_cols:
        if feature not in data.columns:
            continue

        # Get clean data
        clean_data = data[[target_col, feature]].dropna()

        if len(clean_data) < 10:
            continue

        # Test normality of feature
        try:
            _, p_norm = shapiro(clean_data[feature])
            is_normal = p_norm > 0.05
        except:
            is_normal = False

        # Choose appropriate correlation method
        if is_normal:
            stat, p_val = pearsonr(clean_data[target_col], clean_data[feature])
            method = 'Pearson'
        else:
            stat, p_val = spearmanr(clean_data[target_col], clean_data[feature])
            method = 'Spearman'

        correlation_results.append({
            'feature': feature,
            'coefficient': stat,
            'p_value': p_val,
            'method': method,
            'n': len(clean_data),
            'normal': is_normal
        })

    if not correlation_results:
        return []

    # Apply FDR correction for multiple comparisons
    p_values = [r['p_value'] for r in correlation_results]
    reject, p_adj, _, _ = multipletests(p_values, method='fdr_bh', alpha=alpha)

    # Add adjusted p-values and significance
    for i, result in enumerate(correlation_results):
        result['p_adj'] = p_adj[i]
        result['significant'] = reject[i]

    return correlation_results


def cramers_v(chi2, n, r, c):
    """
    Calculate Cramér's V effect size for chi-square test.

    Parameters
    ----------
    chi2 : float
        Chi-square statistic
    n : int
        Total sample size
    r : int
        Number of rows in contingency table
    c : int
        Number of columns in contingency table

    Returns
    -------
    float
        Cramér's V effect size (0 to 1)
        - 0.1: small effect
        - 0.3: medium effect
        - 0.5: large effect

    Examples
    --------
    >>> from scipy.stats import chi2_contingency
    >>> contingency = pd.crosstab(df['Engine'], df['Included'])
    >>> chi2, p, dof, _ = chi2_contingency(contingency)
    >>> v = cramers_v(chi2, contingency.sum().sum(), *contingency.shape)
    >>> print(f"Cramér's V = {v:.3f}")
    """
    return np.sqrt(chi2 / (n * (min(r, c) - 1)))


def cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size for two groups.

    Parameters
    ----------
    group1 : array-like
        First group values
    group2 : array-like
        Second group values

    Returns
    -------
    float
        Cohen's d effect size
        - 0.2: small effect
        - 0.5: medium effect
        - 0.8: large effect

    Examples
    --------
    >>> cited = df[df['cited'] == 1]['word_count']
    >>> not_cited = df[df['cited'] == 0]['word_count']
    >>> d = cohens_d(cited, not_cited)
    >>> print(f"Cohen's d = {d:.3f}")
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def interpret_effect_size(value, measure='d'):
    """
    Interpret effect size magnitude.

    Parameters
    ----------
    value : float
        Effect size value
    measure : str, optional
        Type of effect size: 'd' (Cohen's d), 'v' (Cramér's V), 'r' (correlation)
        By default 'd'

    Returns
    -------
    str
        Interpretation: 'negligible', 'small', 'medium', 'large'

    Examples
    --------
    >>> interpret_effect_size(0.3, 'd')
    'small'
    >>> interpret_effect_size(0.4, 'v')
    'medium'
    """
    abs_value = abs(value)

    if measure == 'd':  # Cohen's d
        if abs_value < 0.2:
            return 'negligible'
        elif abs_value < 0.5:
            return 'small'
        elif abs_value < 0.8:
            return 'medium'
        else:
            return 'large'
    elif measure == 'v':  # Cramér's V
        if abs_value < 0.1:
            return 'negligible'
        elif abs_value < 0.3:
            return 'small'
        elif abs_value < 0.5:
            return 'medium'
        else:
            return 'large'
    elif measure == 'r':  # Correlation coefficient
        if abs_value < 0.1:
            return 'negligible'
        elif abs_value < 0.3:
            return 'small'
        elif abs_value < 0.5:
            return 'medium'
        else:
            return 'large'
    else:
        return 'unknown'


def bootstrap_ci(data, statistic=np.mean, n_bootstrap=10000, alpha=0.05):
    """
    Calculate bootstrap confidence interval for a statistic.

    Parameters
    ----------
    data : array-like
        Sample data
    statistic : callable, optional
        Function to calculate statistic, by default np.mean
    n_bootstrap : int, optional
        Number of bootstrap samples, by default 10000
    alpha : float, optional
        Significance level, by default 0.05

    Returns
    -------
    tuple
        (point_estimate, ci_low, ci_high)

    Examples
    --------
    >>> mean, ci_low, ci_high = bootstrap_ci(df['word_count'])
    >>> print(f"Mean: {mean:.1f} [{ci_low:.1f}, {ci_high:.1f}]")
    """
    data = np.array(data)
    bootstrap_stats = []

    np.random.seed(42)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(statistic(sample))

    point_estimate = statistic(data)
    ci_low = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return point_estimate, ci_low, ci_high
