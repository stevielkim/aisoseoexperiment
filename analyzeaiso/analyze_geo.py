import re, pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


# ------------------------------------------------------
# 1. Load your GEO / AI CSV
# ------------------------------------------------------
AI_CSV_PATH = "ai_serp_analysis.csv"          # update if needed
ai_df = pd.read_csv(AI_CSV_PATH)


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
        LogisticRegression(max_iter=300, solver="liblinear")
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
    print(top_feats)          # Jupyter/IPython – falls back to print if not in notebook
except ValueError as err:
    print(err)

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
