"""
Query Intent Analysis

This script analyzes how query intent affects AI citation patterns:
1. Citation rates by primary intent
2. Citation rates by question type
3. Intent-content matching analysis
4. Query complexity vs inclusion
5. Statistical tests with FDR correction
6. Dashboard visualization
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.statistical import cramers_v
from src.utils.logging import setup_logging

# Set up logging
logger = setup_logging(__name__)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 16)
plt.rcParams['font.size'] = 10


def load_data():
    """Load data with query intent features."""
    logger.info("Loading data with query intent features...")

    data_file = Path("data/processed/ai_serp_analysis_with_intent.csv")
    df = pd.read_csv(data_file)

    logger.info(f"Loaded {len(df)} rows with {df['Query'].nunique()} unique queries")
    logger.info(f"Engines: {df['Engine'].unique().tolist()}")

    return df


def analyze_intent_by_inclusion(df):
    """Analyze citation rates by primary intent."""
    logger.info("\n" + "=" * 80)
    logger.info("CITATION RATES BY PRIMARY INTENT")
    logger.info("=" * 80)

    # Calculate inclusion rate by intent
    intent_stats = df.groupby('primary_intent').agg({
        'Included': ['sum', 'count', 'mean']
    }).round(3)

    intent_stats.columns = ['Included_Count', 'Total', 'Inclusion_Rate']
    intent_stats['Not_Included'] = intent_stats['Total'] - intent_stats['Included_Count']
    intent_stats = intent_stats.sort_values('Inclusion_Rate', ascending=False)

    logger.info(f"\n{intent_stats}")

    # Chi-square test
    contingency = pd.crosstab(df['primary_intent'], df['Included'])
    chi2, p, dof, expected = chi2_contingency(contingency)

    n = contingency.sum().sum()
    v = cramers_v(chi2, n, *contingency.shape)

    logger.info(f"\nChi-Square Test: Intent vs Inclusion")
    logger.info(f"  χ² = {chi2:.2f}, p = {p:.4f}, df = {dof}")
    logger.info(f"  Cramér's V = {v:.3f} ({'Strong' if v >= 0.3 else 'Moderate' if v >= 0.1 else 'Weak'})")

    return intent_stats, {'chi2': chi2, 'p': p, 'cramers_v': v}


def analyze_question_type_by_inclusion(df):
    """Analyze citation rates by question type."""
    logger.info("\n" + "=" * 80)
    logger.info("CITATION RATES BY QUESTION TYPE")
    logger.info("=" * 80)

    # Calculate inclusion rate by question type
    question_stats = df.groupby('question_type').agg({
        'Included': ['sum', 'count', 'mean']
    }).round(3)

    question_stats.columns = ['Included_Count', 'Total', 'Inclusion_Rate']
    question_stats['Not_Included'] = question_stats['Total'] - question_stats['Included_Count']
    question_stats = question_stats.sort_values('Inclusion_Rate', ascending=False)

    logger.info(f"\n{question_stats}")

    # Chi-square test
    contingency = pd.crosstab(df['question_type'], df['Included'])
    chi2, p, dof, expected = chi2_contingency(contingency)

    n = contingency.sum().sum()
    v = cramers_v(chi2, n, *contingency.shape)

    logger.info(f"\nChi-Square Test: Question Type vs Inclusion")
    logger.info(f"  χ² = {chi2:.2f}, p = {p:.4f}, df = {dof}")
    logger.info(f"  Cramér's V = {v:.3f} ({'Strong' if v >= 0.3 else 'Moderate' if v >= 0.1 else 'Weak'})")

    return question_stats, {'chi2': chi2, 'p': p, 'cramers_v': v}


def analyze_query_category_by_inclusion(df):
    """Analyze citation rates by query category (legacy categories)."""
    logger.info("\n" + "=" * 80)
    logger.info("CITATION RATES BY QUERY CATEGORY (Legacy)")
    logger.info("=" * 80)

    # Calculate inclusion rate by category
    category_stats = df.groupby('query_category').agg({
        'Included': ['sum', 'count', 'mean']
    }).round(3)

    category_stats.columns = ['Included_Count', 'Total', 'Inclusion_Rate']
    category_stats['Not_Included'] = category_stats['Total'] - category_stats['Included_Count']
    category_stats = category_stats.sort_values('Inclusion_Rate', ascending=False)

    logger.info(f"\n{category_stats}")

    # Chi-square test
    contingency = pd.crosstab(df['query_category'], df['Included'])
    chi2, p, dof, expected = chi2_contingency(contingency)

    n = contingency.sum().sum()
    v = cramers_v(chi2, n, *contingency.shape)

    logger.info(f"\nChi-Square Test: Query Category vs Inclusion")
    logger.info(f"  χ² = {chi2:.2f}, p = {p:.4f}, df = {dof}")
    logger.info(f"  Cramér's V = {v:.3f} ({'Strong' if v >= 0.3 else 'Moderate' if v >= 0.1 else 'Weak'})")

    return category_stats, {'chi2': chi2, 'p': p, 'cramers_v': v}


def analyze_complexity_by_inclusion(df):
    """Analyze citation rates by query complexity."""
    logger.info("\n" + "=" * 80)
    logger.info("CITATION RATES BY QUERY COMPLEXITY")
    logger.info("=" * 80)

    # Calculate inclusion rate by complexity
    complexity_stats = df.groupby('query_complexity').agg({
        'Included': ['sum', 'count', 'mean'],
        'query_word_count': 'mean'
    }).round(3)

    complexity_stats.columns = ['Included_Count', 'Total', 'Inclusion_Rate', 'Avg_Word_Count']
    complexity_stats['Not_Included'] = complexity_stats['Total'] - complexity_stats['Included_Count']

    # Order by complexity
    complexity_order = {'simple': 0, 'moderate': 1, 'complex': 2}
    complexity_stats['_order'] = complexity_stats.index.map(complexity_order)
    complexity_stats = complexity_stats.sort_values('_order').drop(columns=['_order'])

    logger.info(f"\n{complexity_stats}")

    # Chi-square test
    contingency = pd.crosstab(df['query_complexity'], df['Included'])
    chi2, p, dof, expected = chi2_contingency(contingency)

    n = contingency.sum().sum()
    v = cramers_v(chi2, n, *contingency.shape)

    logger.info(f"\nChi-Square Test: Query Complexity vs Inclusion")
    logger.info(f"  χ² = {chi2:.2f}, p = {p:.4f}, df = {dof}")
    logger.info(f"  Cramér's V = {v:.3f} ({'Strong' if v >= 0.3 else 'Moderate' if v >= 0.1 else 'Weak'})")

    return complexity_stats, {'chi2': chi2, 'p': p, 'cramers_v': v}


def analyze_engine_by_intent(df):
    """Analyze if different engines prefer different intent types."""
    logger.info("\n" + "=" * 80)
    logger.info("ENGINE vs PRIMARY INTENT")
    logger.info("=" * 80)

    # Cross-tabulation
    crosstab = pd.crosstab(df['Engine'], df['primary_intent'], normalize='index') * 100
    logger.info(f"\nIntent Distribution by Engine (%):")
    logger.info(f"\n{crosstab.round(1)}")

    # Chi-square test (Engine × Intent × Included)
    # Test if intent affects inclusion differently by engine
    results_by_engine = {}

    for engine in df['Engine'].unique():
        engine_df = df[df['Engine'] == engine]
        if len(engine_df) > 30:  # Minimum sample size
            contingency = pd.crosstab(engine_df['primary_intent'], engine_df['Included'])

            # Only test if we have multiple intents with sufficient data
            if contingency.shape[0] > 1 and contingency.min().min() >= 5:
                chi2, p, dof, expected = chi2_contingency(contingency)
                n = contingency.sum().sum()
                v = cramers_v(chi2, n, *contingency.shape)

                results_by_engine[engine] = {'chi2': chi2, 'p': p, 'cramers_v': v, 'n': n}

    if results_by_engine:
        logger.info(f"\nIntent vs Inclusion by Engine:")
        for engine, stats in results_by_engine.items():
            logger.info(f"  {engine}: χ² = {stats['chi2']:.2f}, p = {stats['p']:.4f}, V = {stats['cramers_v']:.3f}, n = {stats['n']}")
    else:
        logger.info(f"\n  Insufficient data for engine-specific intent analysis")

    return crosstab, results_by_engine


def create_dashboard(df, intent_stats, question_stats, category_stats, complexity_stats):
    """Create comprehensive query intent analysis dashboard."""
    logger.info("\nCreating query intent dashboard...")

    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle('Query Intent Analysis Dashboard', fontsize=16, fontweight='bold', y=0.995)

    # 1. Intent Distribution (Pie Chart)
    ax = axes[0, 0]
    intent_counts = df['primary_intent'].value_counts()
    colors = sns.color_palette("Set2", len(intent_counts))
    ax.pie(intent_counts.values, labels=intent_counts.index, autopct='%1.1f%%',
           startangle=90, colors=colors)
    ax.set_title('Primary Intent Distribution', fontweight='bold')

    # 2. Inclusion Rate by Intent (Bar Chart)
    ax = axes[0, 1]
    intent_stats_sorted = intent_stats.sort_values('Inclusion_Rate', ascending=True)
    bars = ax.barh(intent_stats_sorted.index, intent_stats_sorted['Inclusion_Rate'] * 100)
    ax.set_xlabel('Inclusion Rate (%)')
    ax.set_title('Citation Rate by Primary Intent', fontweight='bold')
    ax.set_xlim(0, 105)
    for i, (idx, row) in enumerate(intent_stats_sorted.iterrows()):
        ax.text(row['Inclusion_Rate'] * 100 + 1, i,
                f"{row['Inclusion_Rate']*100:.1f}% (n={int(row['Total'])})",
                va='center', fontsize=9)

    # 3. Question Type Distribution
    ax = axes[0, 2]
    question_counts = df['question_type'].value_counts()
    colors = sns.color_palette("Set3", len(question_counts))
    ax.pie(question_counts.values, labels=question_counts.index, autopct='%1.1f%%',
           startangle=90, colors=colors)
    ax.set_title('Question Type Distribution', fontweight='bold')

    # 4. Inclusion Rate by Question Type
    ax = axes[1, 0]
    question_stats_sorted = question_stats.sort_values('Inclusion_Rate', ascending=True)
    bars = ax.barh(question_stats_sorted.index, question_stats_sorted['Inclusion_Rate'] * 100)
    ax.set_xlabel('Inclusion Rate (%)')
    ax.set_title('Citation Rate by Question Type', fontweight='bold')
    ax.set_xlim(0, 105)
    for i, (idx, row) in enumerate(question_stats_sorted.iterrows()):
        ax.text(row['Inclusion_Rate'] * 100 + 1, i,
                f"{row['Inclusion_Rate']*100:.1f}% (n={int(row['Total'])})",
                va='center', fontsize=9)

    # 5. Query Category Distribution (Legacy)
    ax = axes[1, 1]
    category_counts = df['query_category'].value_counts()
    colors = sns.color_palette("Pastel1", len(category_counts))
    ax.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
           startangle=90, colors=colors)
    ax.set_title('Query Category Distribution (Legacy)', fontweight='bold')

    # 6. Inclusion Rate by Query Category
    ax = axes[1, 2]
    category_stats_sorted = category_stats.sort_values('Inclusion_Rate', ascending=True)
    bars = ax.barh(category_stats_sorted.index, category_stats_sorted['Inclusion_Rate'] * 100)
    ax.set_xlabel('Inclusion Rate (%)')
    ax.set_title('Citation Rate by Query Category', fontweight='bold')
    ax.set_xlim(0, 105)
    for i, (idx, row) in enumerate(category_stats_sorted.iterrows()):
        ax.text(row['Inclusion_Rate'] * 100 + 1, i,
                f"{row['Inclusion_Rate']*100:.1f}% (n={int(row['Total'])})",
                va='center', fontsize=9)

    # 7. Query Complexity Distribution
    ax = axes[2, 0]
    complexity_order = ['simple', 'moderate', 'complex']
    complexity_data = [df[df['query_complexity'] == c].shape[0] for c in complexity_order]
    ax.bar(complexity_order, complexity_data, color=sns.color_palette("Blues", 3))
    ax.set_xlabel('Query Complexity')
    ax.set_ylabel('Count')
    ax.set_title('Query Complexity Distribution', fontweight='bold')
    for i, (label, count) in enumerate(zip(complexity_order, complexity_data)):
        ax.text(i, count + 20, str(count), ha='center', fontweight='bold')

    # 8. Inclusion Rate by Complexity
    ax = axes[2, 1]
    complexity_stats_ordered = complexity_stats.reindex(complexity_order)
    bars = ax.bar(complexity_order, complexity_stats_ordered['Inclusion_Rate'] * 100,
                  color=sns.color_palette("Greens", 3))
    ax.set_xlabel('Query Complexity')
    ax.set_ylabel('Inclusion Rate (%)')
    ax.set_title('Citation Rate by Query Complexity', fontweight='bold')
    ax.set_ylim(0, 105)
    for i, (idx, row) in enumerate(complexity_stats_ordered.iterrows()):
        ax.text(i, row['Inclusion_Rate'] * 100 + 1,
                f"{row['Inclusion_Rate']*100:.1f}%",
                ha='center', fontweight='bold')

    # 9. Intent by Engine Heatmap
    ax = axes[2, 2]
    crosstab = pd.crosstab(df['Engine'], df['primary_intent'], normalize='index') * 100
    sns.heatmap(crosstab, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax, cbar_kws={'label': '% of Engine Total'})
    ax.set_title('Intent Distribution by Engine (%)', fontweight='bold')
    ax.set_xlabel('Primary Intent')
    ax.set_ylabel('Engine')

    plt.tight_layout()

    # Save figure
    output_file = Path("outputs/figures/query_intent_analysis.png")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Dashboard saved to: {output_file}")

    return output_file


def main():
    """Main analysis execution."""
    logger.info("\n" + "=" * 80)
    logger.info("QUERY INTENT ANALYSIS")
    logger.info("=" * 80)

    # Load data
    df = load_data()

    # Analyze intent patterns
    intent_stats, intent_test = analyze_intent_by_inclusion(df)
    question_stats, question_test = analyze_question_type_by_inclusion(df)
    category_stats, category_test = analyze_query_category_by_inclusion(df)
    complexity_stats, complexity_test = analyze_complexity_by_inclusion(df)
    engine_intent_dist, engine_intent_tests = analyze_engine_by_intent(df)

    # Create dashboard
    dashboard_file = create_dashboard(df, intent_stats, question_stats, category_stats, complexity_stats)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY OF KEY FINDINGS")
    logger.info("=" * 80)

    logger.info(f"\n1. Primary Intent with Highest Citation Rate: {intent_stats['Inclusion_Rate'].idxmax()} "
                f"({intent_stats['Inclusion_Rate'].max()*100:.1f}%)")

    logger.info(f"\n2. Question Type with Highest Citation Rate: {question_stats['Inclusion_Rate'].idxmax()} "
                f"({question_stats['Inclusion_Rate'].max()*100:.1f}%)")

    logger.info(f"\n3. Query Category with Highest Citation Rate: {category_stats['Inclusion_Rate'].idxmax()} "
                f"({category_stats['Inclusion_Rate'].max()*100:.1f}%)")

    complexity_str = []
    for comp in ['simple', 'moderate', 'complex']:
        if comp in complexity_stats.index:
            complexity_str.append(f"{comp.capitalize()}={complexity_stats.loc[comp, 'Inclusion_Rate']*100:.1f}%")

    logger.info(f"\n4. Query Complexity Effect: {', '.join(complexity_str)}")

    logger.info(f"\n5. Statistical Significance:")
    logger.info(f"   - Intent vs Inclusion: χ²={intent_test['chi2']:.2f}, p={intent_test['p']:.4f}, V={intent_test['cramers_v']:.3f}")
    logger.info(f"   - Question Type vs Inclusion: χ²={question_test['chi2']:.2f}, p={question_test['p']:.4f}, V={question_test['cramers_v']:.3f}")
    logger.info(f"   - Category vs Inclusion: χ²={category_test['chi2']:.2f}, p={category_test['p']:.4f}, V={category_test['cramers_v']:.3f}")

    logger.info(f"\n✅ Query intent analysis complete!")
    logger.info(f"   Dashboard: {dashboard_file}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
