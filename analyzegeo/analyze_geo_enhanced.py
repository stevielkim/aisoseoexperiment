#!/usr/bin/env python3
"""
Enhanced SEO vs AISO Analysis for expanded dataset.
Addresses data quality issues and provides comprehensive insights.
"""
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ------------------------------------------------------
# 1. Load and Clean Data
# ------------------------------------------------------
def load_and_clean_data(csv_path="ai_serp_analysis.csv"):
    """Load and clean the SEO vs AISO dataset."""
    print("📊 Loading and cleaning data...")
    df = pd.read_csv(csv_path)

    # Basic info
    print(f"Raw data: {len(df):,} rows, {len(df.columns)} columns")
    print(f"Engines: {', '.join(df['Engine'].value_counts().index.tolist())}")
    print(f"Date range: {len(df['File'].unique())} unique files")

    # Clean and standardize
    df["Included"] = df["Included"].astype(int)
    if "H1 Count" not in df.columns and "H1 Tags" in df.columns:
        df["H1 Count"] = df["H1 Tags"].fillna("").astype(str).str.split(",").apply(len)
    df["Has H1"] = (df.get("H1 Count", 0) > 0).astype(int)

    # Extract query categories
    df["Query_Type"] = df["File"].str.extract(r'([^/]+)\.html$')[0].str.replace("_", " ")
    df["Query_Category"] = df["Query_Type"].apply(categorize_query)

    return df

def categorize_query(query):
    """Categorize queries for better analysis."""
    query = query.lower()
    if query.startswith(('how to', 'how do', 'how does')):
        return 'How-to'
    elif query.startswith(('what is', 'what are', 'what causes')):
        return 'Informational'
    elif ' vs ' in query or ' versus ' in query:
        return 'Comparison'
    elif query.startswith('best '):
        return 'Best-of'
    elif query.startswith(('benefits of', 'advantages of')):
        return 'Benefits'
    elif 'symptoms of' in query or 'signs of' in query:
        return 'Medical'
    else:
        return 'Other'

# ------------------------------------------------------
# 2. Data Quality Analysis
# ------------------------------------------------------
def analyze_data_quality(df):
    """Analyze data quality and identify issues."""
    print("\n🔍 DATA QUALITY ANALYSIS")
    print("="*50)

    # Engine-specific inclusion rates
    engine_stats = df.groupby('Engine').agg({
        'Included': ['count', 'sum', 'mean'],
        'AI Overview Length': 'mean',
        'Word Count': 'mean'
    }).round(3)

    print("Engine Statistics:")
    print(engine_stats)

    # Identify problematic engines
    inclusion_rates = df.groupby('Engine')['Included'].mean()
    problematic = inclusion_rates[(inclusion_rates > 0.95) | (inclusion_rates < 0.05)]

    if not problematic.empty:
        print("\n⚠️ PROBLEMATIC ENGINES (>95% or <5% inclusion):")
        for engine, rate in problematic.items():
            print(f"  {engine}: {rate:.1%} - {'Too High' if rate > 0.95 else 'Too Low'}")

    return engine_stats

# ------------------------------------------------------
# 3. Enhanced Feature Analysis
# ------------------------------------------------------
def analyze_features_by_engine(df):
    """Analyze what features drive inclusion for each engine separately."""
    print("\n📈 FEATURE ANALYSIS BY ENGINE")
    print("="*50)

    features = ["Word Count", "H1 Count", "H2 Count", "H3 Count",
                "MetaDesc Length", "List Count", "Image Count"]
    features = [f for f in features if f in df.columns]

    results = {}

    for engine in df['Engine'].unique():
        engine_df = df[df['Engine'] == engine].copy()

        # Skip if no variation in inclusion
        if engine_df['Included'].nunique() < 2:
            print(f"\n{engine}: SKIPPED (no variation in inclusion)")
            continue

        print(f"\n{engine}:")
        print(f"  Samples: {len(engine_df):,}")
        print(f"  Inclusion rate: {engine_df['Included'].mean():.1%}")

        # Feature importance using Random Forest
        if len(engine_df) > 50 and engine_df['Included'].mean() not in [0.0, 1.0]:
            X = engine_df[features].fillna(0)
            y = engine_df['Included']

            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)

            importance = pd.DataFrame({
                'Feature': features,
                'Importance': rf.feature_importances_
            }).sort_values('Importance', ascending=False)

            results[engine] = importance
            print("  Top features:")
            for _, row in importance.head(3).iterrows():
                print(f"    {row['Feature']}: {row['Importance']:.3f}")

    return results

# ------------------------------------------------------
# 4. Query Type Analysis
# ------------------------------------------------------
def analyze_query_types(df):
    """Analyze performance by query category."""
    print("\n🎯 QUERY TYPE ANALYSIS")
    print("="*50)

    query_analysis = df.groupby(['Query_Category', 'Engine']).agg({
        'Included': ['count', 'sum', 'mean']
    }).round(3)

    # Flatten column names
    query_analysis.columns = ['_'.join(col).strip() for col in query_analysis.columns]
    query_analysis = query_analysis.reset_index()

    # Pivot for better display
    pivot = query_analysis.pivot(index='Query_Category',
                                columns='Engine',
                                values='Included_mean').fillna(0)

    print("Inclusion rates by Query Type and Engine:")
    print(pivot.round(3))

    return pivot

# ------------------------------------------------------
# 5. Enhanced Visualizations
# ------------------------------------------------------
def create_comprehensive_dashboard(df, feature_importance, query_pivot, save_path="plots/comprehensive_dashboard.png"):
    """Create an enhanced dashboard with multiple insights."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # 1. Engine inclusion rates
    ax1 = fig.add_subplot(gs[0, 0])
    engine_rates = df.groupby('Engine')['Included'].mean()
    bars = ax1.bar(engine_rates.index, engine_rates.values)
    ax1.set_title('Inclusion Rates by Engine', fontweight='bold')
    ax1.set_ylabel('Inclusion Rate')
    ax1.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom')

    # 2. Query type performance
    ax2 = fig.add_subplot(gs[0, 1])
    query_means = df.groupby('Query_Category')['Included'].mean().sort_values(ascending=True)
    ax2.barh(query_means.index, query_means.values, color='lightcoral')
    ax2.set_title('Inclusion by Query Type', fontweight='bold')
    ax2.set_xlabel('Inclusion Rate')

    # 3. Dataset composition
    ax3 = fig.add_subplot(gs[0, 2])
    engine_counts = df['Engine'].value_counts()
    ax3.pie(engine_counts.values, labels=engine_counts.index, autopct='%1.1f%%')
    ax3.set_title('Dataset Composition', fontweight='bold')

    # 4. Rank vs inclusion (overall)
    ax4 = fig.add_subplot(gs[0, 3])
    rank_data = df.groupby('Page Rank')['Included'].mean()
    valid_ranks = rank_data[rank_data.index <= 10]
    ax4.plot(valid_ranks.index, valid_ranks.values, 'o-', color='green')
    ax4.set_title('Inclusion vs Page Rank', fontweight='bold')
    ax4.set_xlabel('Page Rank')
    ax4.set_ylabel('Inclusion Rate')

    # 5-6. Feature importance for top 2 engines (if available)
    engines_with_features = list(feature_importance.keys())[:2]
    for i, engine in enumerate(engines_with_features):
        ax = fig.add_subplot(gs[1, i])
        importance_df = feature_importance[engine].head(6)
        ax.barh(importance_df['Feature'], importance_df['Importance'])
        ax.set_title(f'{engine}: Feature Importance', fontweight='bold')
        ax.set_xlabel('Importance')

    # 7. Heatmap of query performance
    if len(query_pivot.columns) > 1:
        ax7 = fig.add_subplot(gs[1, 2:])
        sns.heatmap(query_pivot, annot=True, fmt='.2f', cmap='RdYlBu_r', ax=ax7)
        ax7.set_title('Query Type Performance Heatmap', fontweight='bold')

    # 8-9. Word count distributions
    ax8 = fig.add_subplot(gs[2, :2])
    for engine in df['Engine'].unique():
        engine_data = df[df['Engine'] == engine]
        ax8.hist(engine_data['Word Count'], alpha=0.6, label=engine, bins=30)
    ax8.set_title('Word Count Distribution by Engine', fontweight='bold')
    ax8.set_xlabel('Word Count')
    ax8.legend()

    # 10. Schema analysis (if available)
    ax10 = fig.add_subplot(gs[2, 2:])
    schema_cols = [col for col in df.columns if 'Schema' in col]
    if schema_cols:
        schema_data = df[schema_cols].sum().sort_values(ascending=True)
        ax10.barh(schema_data.index, schema_data.values)
        ax10.set_title('Schema Markup Usage', fontweight='bold')
        ax10.set_xlabel('Count')
    else:
        ax10.text(0.5, 0.5, 'No Schema Data Available',
                 ha='center', va='center', transform=ax10.transAxes)
        ax10.set_title('Schema Analysis', fontweight='bold')

    plt.suptitle('SEO vs AISO: Comprehensive Analysis Dashboard', fontsize=16, fontweight='bold')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Comprehensive dashboard saved to: {save_path}")
    plt.show()

# ------------------------------------------------------
# 6. Statistical Tests
# ------------------------------------------------------
def statistical_analysis(df):
    """Perform statistical tests on the data."""
    print("\n📊 STATISTICAL ANALYSIS")
    print("="*50)

    # Chi-square test for query type vs inclusion
    from scipy.stats import chi2_contingency

    contingency = pd.crosstab(df['Query_Category'], df['Included'])
    chi2, p_value, dof, expected = chi2_contingency(contingency)

    print(f"Query Type vs Inclusion:")
    print(f"  Chi-square: {chi2:.3f}")
    print(f"  P-value: {p_value:.3f}")
    print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")

    # Engine comparison
    engine_contingency = pd.crosstab(df['Engine'], df['Included'])
    chi2_engine, p_engine, _, _ = chi2_contingency(engine_contingency)

    print(f"\nEngine vs Inclusion:")
    print(f"  Chi-square: {chi2_engine:.3f}")
    print(f"  P-value: {p_engine:.3f}")
    print(f"  Significant: {'Yes' if p_engine < 0.05 else 'No'}")

# ------------------------------------------------------
# 7. Main Execution
# ------------------------------------------------------
def main():
    """Main analysis pipeline."""
    print("🚀 SEO vs AISO Enhanced Analysis")
    print("="*50)

    # Load and clean data
    df = load_and_clean_data()

    # Data quality analysis
    quality_stats = analyze_data_quality(df)

    # Feature analysis by engine
    feature_importance = analyze_features_by_engine(df)

    # Query type analysis
    query_pivot = analyze_query_types(df)

    # Statistical tests
    statistical_analysis(df)

    # Create comprehensive dashboard
    import os
    os.makedirs("plots", exist_ok=True)
    create_comprehensive_dashboard(df, feature_importance, query_pivot)

    # Summary insights
    print("\n🎯 KEY INSIGHTS")
    print("="*50)
    total_queries = len(df['Query_Type'].unique())
    total_engines = len(df['Engine'].unique())
    overall_inclusion = df['Included'].mean()

    print(f"📈 Dataset: {len(df):,} results across {total_queries} queries and {total_engines} engines")
    print(f"📊 Overall inclusion rate: {overall_inclusion:.1%}")
    print(f"🏆 Best performing query type: {df.groupby('Query_Category')['Included'].mean().idxmax()}")
    print(f"🔧 Most reliable engine: {df.groupby('Engine')['Included'].std().idxmin()}")

if __name__ == "__main__":
    main()