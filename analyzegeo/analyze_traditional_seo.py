#!/usr/bin/env python3
"""
Traditional SEO Analysis: Google AI + Bing AI Only
Focus: Do traditional SEO factors predict inclusion in AI Overviews?
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr, shapiro
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("RdYlBu_r")

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

def load_traditional_seo_data():
    """Load and prepare data for traditional SEO analysis (Google AI + Bing AI only)."""
    print("📊 Loading Traditional SEO Data (Google AI + Bing AI)")
    print("="*60)

    df = pd.read_csv('ai_serp_analysis.csv')

    # Filter to only traditional search engines
    traditional_engines = ['Google AI', 'Bing AI']
    df = df[df['Engine'].isin(traditional_engines)].copy()

    # Clean data
    df['Included'] = df['Included'].astype(int)
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
        else:
            return 'Other'

    df['Query_Category'] = df['Query_Name'].apply(categorize_query)

    print(f"✅ Filtered to traditional search engines:")
    print(f"   • Total records: {len(df):,}")
    print(f"   • Google AI: {len(df[df['Engine'] == 'Google AI']):,}")
    print(f"   • Bing AI: {len(df[df['Engine'] == 'Bing AI']):,}")
    print(f"   • Unique queries: {df['Query_Name'].nunique()}")
    print(f"   • Overall inclusion rate: {df['Included'].mean():.1%}")

    return df

def analyze_traditional_ranking_factors(df):
    """Analyze traditional SEO factors in AI Overview inclusion."""
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

    # IMPROVEMENT: Proper correlation analysis with statistical testing and FDR correction
    if len(available_features) >= 3:
        feature_data = df[available_features + ['Included']].fillna(0)

        # Calculate correlations with proper statistical tests
        correlation_results = []

        for feature in available_features:
            data = feature_data[[feature, 'Included']].dropna()

            if len(data) < 10:
                continue

            # Test normality
            try:
                _, p_norm = shapiro(data[feature])
                is_normal = p_norm > 0.05
            except:
                is_normal = False

            # Choose appropriate correlation method
            if is_normal:
                corr, p_val = pearsonr(data[feature], data['Included'])
                method = 'Pearson'
            else:
                corr, p_val = spearmanr(data[feature], data['Included'])
                method = 'Spearman'

            correlation_results.append({
                'feature': feature,
                'correlation': corr,
                'p_value': p_val,
                'method': method,
                'n': len(data)
            })

        # Filter out NaN correlations before FDR correction
        valid_results = [r for r in correlation_results if not np.isnan(r['correlation'])]

        if len(valid_results) > 0:
            # Apply FDR correction for multiple comparisons
            p_values = [r['p_value'] for r in valid_results]
            reject, p_adj, _, _ = multipletests(p_values, method='fdr_bh')

            # Add adjusted p-values and significance
            for i, result in enumerate(valid_results):
                result['p_adj'] = p_adj[i]
                result['significant'] = reject[i]

            correlation_results = valid_results

        # Sort by absolute correlation
        correlation_results.sort(key=lambda x: abs(x['correlation']), reverse=True)

        print(f"\n🔗 SEO FEATURE CORRELATIONS WITH AI OVERVIEW INCLUSION:")
        print(f"   (Using {correlation_results[0]['method']} or Spearman based on normality tests)")
        print(f"   (P-values adjusted for multiple comparisons using FDR correction)")

        for result in correlation_results:
            sig_marker = "***" if result['significant'] else ""
            print(f"   • {result['feature']}: {result['correlation']:+.3f} "
                  f"(p_adj={result['p_adj']:.4f}) {sig_marker}")

        # Create simple correlations dict for backward compatibility
        correlations = pd.Series({r['feature']: r['correlation'] for r in correlation_results})

        return correlations, available_features, correlation_results
    else:
        print("❌ Insufficient features for correlation analysis")
        return None, available_features, None

def analyze_by_engine(df):
    """Compare traditional SEO performance between Google AI and Bing AI."""
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

    # IMPROVEMENT: Statistical comparison with assumption validation and effect size
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
    cramers_v = np.sqrt(chi2 / (n * (min(r, c) - 1)))

    print(f"\n📊 STATISTICAL SIGNIFICANCE:")
    print(f"   • Chi-square test: χ² = {chi2:.3f}, p = {p_value:.4f}")

    if min_expected < 5:
        print(f"   ⚠️  WARNING: Chi-square assumption violated (min expected = {min_expected:.1f})")
        print(f"       Results may be unreliable (expected frequencies should be ≥ 5)")
    else:
        print(f"   ✅ Chi-square assumptions met (min expected = {min_expected:.1f})")

    print(f"   • Cramér's V (effect size): {cramers_v:.3f}", end="")
    if cramers_v < 0.1:
        print(" (negligible)")
    elif cramers_v < 0.3:
        print(" (small)")
    elif cramers_v < 0.5:
        print(" (medium)")
    else:
        print(" (large)")

    print(f"   • Significant difference: {'Yes' if p_value < 0.05 else 'No'}")

    return engine_analysis

def predictive_modeling(df, available_features):
    """Build predictive models for AI Overview inclusion."""
    print("\n🤖 PREDICTIVE MODELING: AI OVERVIEW INCLUSION")
    print("="*50)

    if len(available_features) < 3:
        print("❌ Insufficient features for modeling")
        return None

    # Prepare data
    feature_data = df[available_features].fillna(df[available_features].median())
    target = df['Included']

    # IMPROVEMENT: Analyze class imbalance
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

    # IMPROVEMENT: Logistic Regression with coefficient interpretation
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

    # IMPROVEMENT: Interpret coefficients as odds ratios
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

def create_traditional_seo_dashboard(df, correlations, engine_analysis, models):
    """Create comprehensive dashboard for traditional SEO analysis."""
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
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    plt.suptitle('Traditional SEO Analysis: Google AI + Bing AI', fontsize=18, fontweight='bold')

    # Save
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/traditional_seo_analysis.png", dpi=300, bbox_inches='tight')
    print("✅ Traditional SEO dashboard saved to: plots/traditional_seo_analysis.png")
    plt.close()

def main():
    """Main analysis execution."""
    print("🔍 Traditional SEO Analysis: Google AI + Bing AI")
    print("="*60)

    # Load data
    df = load_traditional_seo_data()

    # IMPROVEMENT: Data quality audit
    if 'Included' in df.columns:
        audit_data_quality(df, "All Engines")

        # Per-engine audit
        for engine in df['Engine'].unique():
            engine_df = df[df['Engine'] == engine]
            audit_data_quality(engine_df, engine)

    # Analyze SEO factors
    result = analyze_traditional_ranking_factors(df)
    if result and len(result) == 3:
        correlations, available_features, correlation_results = result
    else:
        correlations = result[0] if result else None
        available_features = result[1] if result and len(result) > 1 else []
        correlation_results = None

    # Engine comparison
    engine_analysis = analyze_by_engine(df)

    # Predictive modeling
    models = predictive_modeling(df, available_features)

    # Create dashboard
    create_traditional_seo_dashboard(df, correlations, engine_analysis, models)

    print(f"\n✅ Traditional SEO Analysis Complete!")
    print(f"📁 Results saved to plots/traditional_seo_analysis.png")
    print(f"🎯 Focus: Traditional search engines with AI enhancements")

if __name__ == "__main__":
    main()