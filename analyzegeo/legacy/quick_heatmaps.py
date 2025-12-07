#!/usr/bin/env python3
"""
Quick heatmap generation without display issues.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
plt.style.use('default')
sns.set_palette("RdYlBu_r")

def main():
    print("🔥 Creating SEO vs AISO Heatmaps...")

    # Load data
    df = pd.read_csv('ai_serp_analysis.csv')
    df['Included'] = df['Included'].astype(int)
    df['Query_Name'] = df['File'].str.extract(r'([^/]+)\.html$')[0].str.replace('_', ' ')

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

    # Create plots directory
    os.makedirs("plots", exist_ok=True)

    # 1. Engine vs Query Category Heatmap
    print("📊 Creating Engine-Query heatmap...")
    heatmap_data = df.groupby(['Query_Category', 'Engine'])['Included'].mean().unstack(fill_value=0)

    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data,
                annot=True,
                fmt='.2f',
                cmap='RdYlBu_r',
                center=0.5,
                square=True,
                linewidths=0.5)
    plt.title('Inclusion Rates: Query Category vs Engine', fontweight='bold')
    plt.tight_layout()
    plt.savefig("plots/engine_query_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Engine-Query heatmap saved")

    # 2. Feature Correlation Heatmap
    print("📊 Creating Feature correlation heatmap...")
    feature_cols = ['Word Count', 'H1 Count', 'H2 Count', 'H3 Count',
                   'MetaDesc Length', 'Page Rank', 'Included']
    available_cols = [col for col in feature_cols if col in df.columns]

    if len(available_cols) >= 3:
        correlation_data = df[available_cols].corr()

        plt.figure(figsize=(8, 6))
        mask = np.triu(np.ones_like(correlation_data, dtype=bool))
        sns.heatmap(correlation_data,
                    annot=True,
                    fmt='.2f',
                    cmap='RdBu_r',
                    center=0,
                    mask=mask,
                    linewidths=0.5)
        plt.title('SEO Feature Correlation Matrix', fontweight='bold')
        plt.tight_layout()
        plt.savefig("plots/feature_correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Feature correlation heatmap saved")

    # 3. Rank Performance Heatmap
    print("📊 Creating Rank performance heatmap...")
    rank_data = df[df['Page Rank'] <= 10].groupby(['Page Rank', 'Engine'])['Included'].mean().unstack(fill_value=0)

    plt.figure(figsize=(8, 8))
    sns.heatmap(rank_data,
                annot=True,
                fmt='.2f',
                cmap='RdYlBu_r',
                center=0.5,
                linewidths=0.5)
    plt.title('Inclusion Rate by Page Rank and Engine', fontweight='bold')
    plt.tight_layout()
    plt.savefig("plots/rank_performance_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Rank performance heatmap saved")

    # Summary stats
    print("\n🎯 KEY INSIGHTS:")
    print(f"📈 Best query category: {heatmap_data.mean(axis=1).idxmax()}")
    print(f"📉 Worst query category: {heatmap_data.mean(axis=1).idxmin()}")
    print(f"🏆 Best engine overall: {heatmap_data.mean(axis=0).idxmax()}")

    if len(available_cols) >= 3:
        inclusion_corr = correlation_data['Included'].abs().sort_values(ascending=False)
        print(f"🔗 Feature most correlated with inclusion: {inclusion_corr.index[1]} ({inclusion_corr.iloc[1]:.3f})")

    print("\n✅ All heatmaps saved to plots/ directory!")

if __name__ == "__main__":
    main()