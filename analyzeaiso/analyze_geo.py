import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import os


# ------------------------------------------------------
# 1. Load your GEO / AI CSV
# ------------------------------------------------------
AI_CSV_PATH = "ai_serp_analysis.csv"          # update if needed
ai_df = pd.read_csv(AI_CSV_PATH)
ai_df["Included"] = ai_df["Included"].astype(int)   # True → 1, False → 0


# Ensure numeric count exists (if you ever ingest rows without it)
if "H1 Count" not in ai_df.columns:
    ai_df["H1 Count"] = (
        ai_df["H1 Tags"].fillna("").astype(str).str.split(",").apply(len)
    )

# Create / overwrite binary flag
ai_df["Has H1"] = (ai_df["H1 Count"] > 0).astype(int)


# ------------------------------------------------------
# 2. Quick sanity check – make sure Included & features exist
# ------------------------------------------------------
required_cols = [
    "Included", "Word Count", "Has H1", "H2 Count", "H3 Count",
    "MetaDesc Length", "Snippet Length"
]
# removed "AI Overview Length" but can use to describe 
# whether long AI answers pull from more sources vs short answers
# ai.groupby(pd.qcut(ai["AI Overview Length"], 4))["Included"].mean()

missing = [c for c in required_cols if c not in ai_df.columns]
if missing:
    raise ValueError(f"Missing columns in {AI_CSV_PATH}: {missing}")

print("Included value counts:", ai_df["Included"].value_counts(dropna=False))

# ------------------------------------------------------
# 3. Function – (copy from previous message or import)
# ------------------------------------------------------


def top_inclusion_features(ai_df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    if ai_df["Included"].nunique() < 2:
        raise ValueError("Included flag has only one class (0/1); can't train model.")
    
    num_cols = [
        "Word Count","H1 Count","H2 Count","H3 Count",
        "MetaDesc Length","Snippet Length"
    ]
    num_cols = [c for c in num_cols if c in ai_df.columns]
    
    X = ai_df[num_cols].fillna(0)
    y = ai_df["Included"]
    
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
    )
    pipe.fit(X, y)
    coefs = pipe.named_steps["logisticregression"].coef_[0]
    
    out = (
        pd.DataFrame({"Feature": num_cols, "Coeff": coefs})
        .sort_values("Coeff", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    out["Odds-multiplier"] = np.exp(out["Coeff"]).round(2)
    return out

# ------------------------------------------------------
# 4. Run and display top-N inclusion drivers
# ------------------------------------------------------
top_n = 8
try:
    top_feats = top_inclusion_features(ai_df, n=top_n)
    print(f"\nTop {top_n} features that raise odds of inclusion:")
    print(top_feats)          
except ValueError as err:
    print(err)

def plot_ai_inclusion_dashboard(df, feat_table, save_path="ai_inclusion_dashboard.png"):
    """Create and save AI inclusion analysis dashboard."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1) Odds-ratio
    feat_sorted = feat_table.sort_values("Odds-multiplier")
    axes[0].barh(feat_sorted["Feature"], feat_sorted["Odds-multiplier"], color="#4c72b0")
    axes[0].axvline(1, ls="--", c="k")
    axes[0].set_title("Odds ratio")
    axes[0].set_xlabel("Odds-multiplier")
    
    # 2) Rank bucket lift
    buckets = pd.cut(df["Page Rank"], [0, 1, 3, 10, np.inf], labels=["1", "2–3", "4–10", "11+"])
    rank_data = df.groupby(buckets, observed=True)["Included"].mean().mul(100)
    axes[1].bar(range(len(rank_data)), rank_data.values, color="#dd8452")
    axes[1].set_xticks(range(len(rank_data)))
    axes[1].set_xticklabels(rank_data.index, rotation=0)
    axes[1].set_ylabel("% cited")
    axes[1].set_title("Baseline by rank")
    
    # 3) Word-count violin
    sns.violinplot(x="Included", y="Word Count", data=df, inner="quartile",
                   hue="Included", palette="Pastel1", ax=axes[2], legend=False)
    axes[2].set_xticks([0, 1])
    axes[2].set_xticklabels(["No", "Yes"])
    axes[2].set_title("Word count spread")
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Dashboard saved to: {save_path}")
    plt.show()


# ------------------------------------------------------
# Page-Rank vs Inclusion: descriptive and correlation view
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

    print(f"\nCorrelation (Page Rank vs Included): "
          f"Pearson r = {pear:.3f}, Spearman ρ = {spear:.3f}")
    return g

# ---------- run it & display --------------------------------------
rank_summary = rank_vs_inclusion(ai_df, max_rank=10)
print("\nPage-Rank bucket summary (Top 10 ranks):")
print(rank_summary)   # use print(rank_summary) outside Jupyter
print(top_feats)
print("Columns in top_feats:", top_feats.columns.tolist())

# Create output directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Generate and save the dashboard
plot_ai_inclusion_dashboard(ai_df, top_feats, save_path="plots/ai_inclusion_dashboard.png")
print(ai_df["Included"].value_counts())   # should output something like 320 zeros, 35 ones
