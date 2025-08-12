import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ------------------------------------------------------
# 1. Load your GEO / AI CSV
# ------------------------------------------------------
AI_CSV_PATH = "ai_serp_analysis.csv"
logger.info(f"Loading data from {AI_CSV_PATH}")

try:
    ai_df = pd.read_csv(AI_CSV_PATH)
    logger.info(f"Loaded {len(ai_df)} rows with {len(ai_df.columns)} columns")
except FileNotFoundError:
    logger.error(f"File {AI_CSV_PATH} not found. Please run parse_geo.py first.")
    raise

# Convert Included to int and handle any boolean values
ai_df["Included"] = ai_df["Included"].astype(int)   # True → 1, False → 0

# Data quality check
logger.info(f"Data shape: {ai_df.shape}")
logger.info(f"Columns available: {list(ai_df.columns)}")


# ------------------------------------------------------
# 2. Quick sanity check – make sure Included & features exist
# ------------------------------------------------------
required_cols = [
    "Included", "Word Count", "H1 Count", "H2 Count", "H3 Count",
    "MetaDesc Length", "Snippet Length"
]

missing = [c for c in required_cols if c not in ai_df.columns]
if missing:
    logger.error(f"Missing columns in {AI_CSV_PATH}: {missing}")
    raise ValueError(f"Missing columns in {AI_CSV_PATH}: {missing}")

logger.info("Included value counts:")
logger.info(ai_df["Included"].value_counts(dropna=False))

# ------------------------------------------------------
# 3. Function – (copy from previous message or import)
# ------------------------------------------------------


def top_inclusion_features(ai_df: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    """Enhanced feature analysis with new structural and schema features."""
    if ai_df["Included"].nunique() < 2:
        raise ValueError("Included flag has only one class (0/1); can't train model.")
    
    # Expanded feature set including new structural and schema features
    num_cols = [
        # Original features
        "Word Count", "H1 Count", "H2 Count", "H3 Count",
        "MetaDesc Length", "Snippet Length", "AI Overview Length",
        
        # New structural features
        "List Count", "OL Count", "UL Count", "List Item Count", 
        "Table Count", "Avg Paragraph Length", "Short Paragraphs (<150)",
        "Image Count", "Images Alt>=50", "Question Heading Count", "Has TOC Anchors",
        
        # Schema features
        "Has FAQ Schema", "Has HowTo Schema", "Has Article Schema"
    ]
    
    # Filter to only include columns that exist in the dataset
    available_cols = [c for c in num_cols if c in ai_df.columns]
    logger.info(f"Using {len(available_cols)} features for analysis: {available_cols}")
    
    X = ai_df[available_cols].fillna(0)
    y = ai_df["Included"]
    
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
    )
    pipe.fit(X, y)
    coefs = pipe.named_steps["logisticregression"].coef_[0]
    
    out = (
        pd.DataFrame({"Feature": available_cols, "Coeff": coefs})
        .sort_values("Coeff", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    out["Odds-multiplier"] = np.exp(out["Coeff"]).round(2)
    return out

# ------------------------------------------------------
# 4. Run and display top-N inclusion drivers
# ------------------------------------------------------
top_n = 12
try:
    top_feats = top_inclusion_features(ai_df, n=top_n)
    logger.info(f"\nTop {top_n} features that raise odds of inclusion:")
    logger.info(f"\n{top_feats}")
    print(f"\nTop {top_n} features that raise odds of inclusion:")
    print(top_feats)          
except ValueError as err:
    logger.error(f"Error in feature analysis: {err}")
    print(err)

def plot_ai_inclusion_dashboard(df, feat_table, save_path="ai_inclusion_dashboard.png"):
    """Enhanced dashboard with new structural and schema features."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1) Odds-ratio (top features)
    feat_sorted = feat_table.sort_values("Odds-multiplier")
    axes[0,0].barh(feat_sorted["Feature"], feat_sorted["Odds-multiplier"], color="#4c72b0")
    axes[0,0].axvline(1, ls="--", c="k")
    axes[0,0].set_title("Top Features: Odds Ratio")
    axes[0,0].set_xlabel("Odds-multiplier")
    
    # 2) Rank bucket lift
    buckets = pd.cut(df["Page Rank"], [0, 1, 3, 10, np.inf], labels=["1", "2–3", "4–10", "11+"])
    rank_data = df.groupby(buckets, observed=True)["Included"].mean().mul(100)
    axes[0,1].bar(range(len(rank_data)), rank_data.values, color="#dd8452")
    axes[0,1].set_xticks(range(len(rank_data)))
    axes[0,1].set_xticklabels(rank_data.index, rotation=0)
    axes[0,1].set_ylabel("% cited")
    axes[0,1].set_title("Inclusion Rate by Rank")
    
    # 3) Word-count violin
    sns.violinplot(x="Included", y="Word Count", data=df, inner="quartile",
                   hue="Included", palette="Pastel1", ax=axes[0,2], legend=False)
    axes[0,2].set_xticks([0, 1])
    axes[0,2].set_xticklabels(["No", "Yes"])
    axes[0,2].set_title("Word Count Distribution")
    
    # 4) Schema features analysis
    schema_cols = ["Has FAQ Schema", "Has HowTo Schema", "Has Article Schema"]
    schema_data = []
    for col in schema_cols:
        if col in df.columns:
            inclusion_rate = df.groupby(col)["Included"].mean().mul(100)
            schema_data.append({
                'Schema': col.replace('Has ', '').replace(' Schema', ''),
                'No Schema': inclusion_rate.get(0, 0),
                'With Schema': inclusion_rate.get(1, 0)
            })
    
    if schema_data:
        schema_df = pd.DataFrame(schema_data)
        schema_df.plot(x='Schema', y=['No Schema', 'With Schema'], 
                      kind='bar', ax=axes[1,0], color=['#ff9999', '#66b3ff'])
        axes[1,0].set_title("Inclusion Rate by Schema Type")
        axes[1,0].set_ylabel("% Included")
        axes[1,0].legend()
    
    # 5) Structural features
    struct_features = ["List Count", "Table Count", "Image Count"]
    struct_data = []
    for feat in struct_features:
        if feat in df.columns:
            # Create binary feature (high/low)
            median_val = df[feat].median()
            df[f"{feat}_binary"] = (df[feat] > median_val).astype(int)
            inclusion_rate = df.groupby(f"{feat}_binary")["Included"].mean().mul(100)
            struct_data.append({
                'Feature': feat.replace(' Count', ''),
                'Low': inclusion_rate.get(0, 0),
                'High': inclusion_rate.get(1, 0)
            })
    
    if struct_data:
        struct_df = pd.DataFrame(struct_data)
        struct_df.plot(x='Feature', y=['Low', 'High'], 
                      kind='bar', ax=axes[1,1], color=['#ffcc99', '#99ff99'])
        axes[1,1].set_title("Inclusion Rate by Structural Features")
        axes[1,1].set_ylabel("% Included")
        axes[1,1].legend()
    
    # 6) Engine comparison
    if "Engine" in df.columns:
        engine_data = df.groupby("Engine")["Included"].mean().mul(100)
        engine_data.plot(kind='bar', ax=axes[1,2], color='#c5c5c5')
        axes[1,2].set_title("Inclusion Rate by Search Engine")
        axes[1,2].set_ylabel("% Included")
        axes[1,2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Enhanced dashboard saved to: {save_path}")
    plt.show()


# ------------------------------------------------------
# Enhanced Analysis Functions
# ------------------------------------------------------
def rank_vs_inclusion(ai_df, max_rank=20):
    """
    Returns a DataFrame:
        Page Rank | # Results | # Included | % Included
    plus prints Pearson & Spearman correlations between
    Page Rank (numeric) and Included (0/1).
    """
    # ensure numeric
    tmp = ai_df[["Page Rank", "Included"]].dropna()
    tmp["Page Rank"] = pd.to_numeric(tmp["Page Rank"], errors="coerce")

    # descriptive table
    g = (
        tmp[tmp["Page Rank"] <= max_rank]
        .groupby("Page Rank")["Included"]
        .agg(total="count", included="sum")
        .reset_index()
    )
    g["Pct Included"] = (g["included"] / g["total"] * 100).round(1)

    # correlations (note: higher rank number = worse position)
    pear = tmp["Page Rank"].corr(tmp["Included"])          # point-biserial/pearson
    spear= tmp["Page Rank"].corr(tmp["Included"], method="spearman")

    logger.info(f"Correlation (Page Rank vs Included): "
          f"Pearson r = {pear:.3f}, Spearman ρ = {spear:.3f}")
    return g

def analyze_schema_impact(ai_df):
    """Analyze the impact of schema markup on inclusion rates."""
    schema_cols = ["Has FAQ Schema", "Has HowTo Schema", "Has Article Schema"]
    schema_analysis = {}
    
    for col in schema_cols:
        if col in ai_df.columns:
            # Calculate inclusion rates
            rates = ai_df.groupby(col)["Included"].agg(['count', 'sum', 'mean'])
            rates['pct_included'] = (rates['mean'] * 100).round(1)
            
            # Chi-square test for independence
            from scipy.stats import chi2_contingency
            contingency = pd.crosstab(ai_df[col], ai_df["Included"])
            if contingency.shape == (2, 2):
                chi2, p_value, dof, expected = chi2_contingency(contingency)
                schema_analysis[col] = {
                    'rates': rates,
                    'chi2': chi2,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
    
    return schema_analysis

def analyze_structural_features(ai_df):
    """Analyze the impact of structural features on inclusion."""
    struct_features = [
        "List Count", "Table Count", "Image Count", "Question Heading Count",
        "Has TOC Anchors", "Avg Paragraph Length"
    ]
    
    struct_analysis = {}
    for feat in struct_features:
        if feat in ai_df.columns:
            # Create binary feature (high/low based on median)
            median_val = ai_df[feat].median()
            ai_df[f"{feat}_binary"] = (ai_df[feat] > median_val).astype(int)
            
            # Calculate inclusion rates
            rates = ai_df.groupby(f"{feat}_binary")["Included"].agg(['count', 'sum', 'mean'])
            rates['pct_included'] = (rates['mean'] * 100).round(1)
            
            struct_analysis[feat] = {
                'median': median_val,
                'rates': rates,
                'correlation': ai_df[feat].corr(ai_df["Included"])
            }
    
    return struct_analysis

def analyze_engine_differences(ai_df):
    """Analyze differences between search engines."""
    if "Engine" not in ai_df.columns:
        return None
    
    engine_analysis = {}
    
    # Overall inclusion rates by engine
    engine_rates = ai_df.groupby("Engine")["Included"].agg(['count', 'sum', 'mean'])
    engine_rates['pct_included'] = (engine_rates['mean'] * 100).round(1)
    engine_analysis['overall_rates'] = engine_rates
    
    # Feature differences by engine
    feature_cols = ["Word Count", "H1 Count", "H2 Count", "MetaDesc Length", "Snippet Length"]
    engine_features = {}
    
    for feat in feature_cols:
        if feat in ai_df.columns:
            engine_features[feat] = ai_df.groupby("Engine")[feat].agg(['mean', 'std']).round(2)
    
    engine_analysis['feature_means'] = engine_features
    
    return engine_analysis

# ---------- run enhanced analyses & display --------------------------------------
logger.info("Running enhanced AI SERP analysis...")

# 1. Rank vs Inclusion analysis
rank_summary = rank_vs_inclusion(ai_df, max_rank=10)
logger.info("\nPage-Rank bucket summary (Top 10 ranks):")
logger.info(f"\n{rank_summary}")

# 2. Schema impact analysis
logger.info("\nAnalyzing schema markup impact...")
schema_analysis = analyze_schema_impact(ai_df)
if schema_analysis:
    for schema_type, analysis in schema_analysis.items():
        logger.info(f"\n{schema_type}:")
        logger.info(f"  Rates: {analysis['rates']['pct_included'].to_dict()}")
        logger.info(f"  Chi-square: {analysis['chi2']:.3f}, p-value: {analysis['p_value']:.3f}")
        logger.info(f"  Significant: {analysis['significant']}")

# 3. Structural features analysis
logger.info("\nAnalyzing structural features...")
struct_analysis = analyze_structural_features(ai_df)
if struct_analysis:
    for feature, analysis in struct_analysis.items():
        logger.info(f"\n{feature}:")
        logger.info(f"  Median: {analysis['median']}")
        logger.info(f"  Inclusion rates: {analysis['rates']['pct_included'].to_dict()}")
        logger.info(f"  Correlation with inclusion: {analysis['correlation']:.3f}")

# 4. Engine differences analysis
logger.info("\nAnalyzing search engine differences...")
engine_analysis = analyze_engine_differences(ai_df)
if engine_analysis:
    logger.info(f"\nEngine inclusion rates:\n{engine_analysis['overall_rates']['pct_included']}")

# Create output directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Generate and save the enhanced dashboard
logger.info("\nGenerating enhanced dashboard...")
plot_ai_inclusion_dashboard(ai_df, top_feats, save_path="plots/ai_inclusion_dashboard.png")

# Print summary statistics
logger.info(f"\nFinal summary:")
logger.info(f"Total results: {len(ai_df)}")
logger.info(f"Inclusion counts: {ai_df['Included'].value_counts().to_dict()}")
logger.info(f"Engines analyzed: {ai_df['Engine'].unique().tolist() if 'Engine' in ai_df.columns else 'N/A'}")

# Display key results in console
print(f"\n{'='*60}")
print("ENHANCED AI SERP ANALYSIS RESULTS")
print(f"{'='*60}")
print(f"Total results analyzed: {len(ai_df)}")
print(f"Inclusion rate: {(ai_df['Included'].mean()*100):.1f}%")
print(f"Top features affecting inclusion:")
print(top_feats[['Feature', 'Odds-multiplier']].head(5))
print(f"\nDashboard saved to: plots/ai_inclusion_dashboard.png")
print(f"{'='*60}")
