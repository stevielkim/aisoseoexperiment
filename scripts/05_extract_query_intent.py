"""
Extract query intent features from existing data.

This script:
1. Reads the ai_serp_analysis.csv file
2. Extracts unique queries from filenames
3. Classifies each query using QueryIntentClassifier
4. Saves query intent features to CSV
5. Merges intent features back into source_features.csv
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.query_intent import QueryIntentClassifier


def extract_query_from_filename(filename: str) -> str:
    """
    Extract query text from filename.

    Example:
        'perplexity_search_results_html/symptoms_of_vitamin_D_deficiency_perplexity.html'
        -> 'symptoms of vitamin D deficiency'

    Parameters:
    -----------
    filename : str
        File path from CSV

    Returns:
    --------
    str : Extracted query text
    """
    # Get basename
    basename = Path(filename).stem

    # Remove engine suffix (_perplexity, _google, _bing)
    for engine in ['_perplexity', '_google', '_bing', '_copilot']:
        if basename.endswith(engine):
            basename = basename[:-len(engine)]
            break

    # Replace underscores with spaces
    query = basename.replace('_', ' ')

    return query


def main():
    """Main execution function."""
    print("=" * 80)
    print("QUERY INTENT FEATURE EXTRACTION")
    print("=" * 80)

    # Paths
    input_file = Path("data/processed/ai_serp_analysis.csv")
    output_file = Path("data/processed/query_intent_features.csv")

    print(f"\n1. Loading data from: {input_file}")

    # Load data
    df = pd.read_csv(input_file)
    print(f"   Loaded {len(df)} rows")

    # Extract unique queries
    print(f"\n2. Extracting unique queries from File column...")
    df['Query'] = df['File'].apply(extract_query_from_filename)
    unique_queries = df['Query'].unique()
    print(f"   Found {len(unique_queries)} unique queries")

    # Sample queries
    print(f"\n   Sample queries:")
    for q in list(unique_queries)[:5]:
        print(f"     - {q}")

    # Initialize classifier
    print(f"\n3. Classifying query intent...")
    classifier = QueryIntentClassifier()

    # Classify each query
    query_intent_data = []
    for query in unique_queries:
        features = classifier.classify(query)
        features['query'] = query
        features['query_category'] = classifier.categorize_query(query)
        query_intent_data.append(features)

    # Create DataFrame
    intent_df = pd.DataFrame(query_intent_data)

    # Reorder columns (query first)
    cols = ['query'] + [c for c in intent_df.columns if c != 'query']
    intent_df = intent_df[cols]

    print(f"   Classified {len(intent_df)} queries")

    # Show intent distribution
    print(f"\n4. Intent Distribution:")
    print(intent_df['primary_intent'].value_counts())

    print(f"\n   Question Type Distribution:")
    print(intent_df['question_type'].value_counts())

    print(f"\n   Query Category Distribution (Legacy):")
    print(intent_df['query_category'].value_counts())

    # Save to CSV
    print(f"\n5. Saving query intent features to: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    intent_df.to_csv(output_file, index=False)
    print(f"   Saved {len(intent_df)} query intent records")

    # Merge with ai_serp_analysis.csv (which has File and Query columns)
    print(f"\n6. Merging intent features with ai_serp_analysis.csv...")

    # Merge back with original df (which already has Query column)
    merged_df = df.merge(
        intent_df,
        left_on='Query',
        right_on='query',
        how='left',
        suffixes=('', '_intent')
    )

    # Drop duplicate query column if it exists
    if 'query' in merged_df.columns:
        merged_df = merged_df.drop(columns=['query'])

    # Save enhanced ai_serp_analysis
    enhanced_file = Path("data/processed/ai_serp_analysis_with_intent.csv")
    merged_df.to_csv(enhanced_file, index=False)
    print(f"   Saved enhanced analysis to: {enhanced_file}")
    print(f"   Total rows: {len(merged_df)}")

    # Show sample
    print(f"\n   Sample merged data:")
    intent_cols = ['Query', 'Engine', 'Included', 'primary_intent', 'question_type', 'query_category']
    available_cols = [c for c in intent_cols if c in merged_df.columns]
    print(merged_df[available_cols].head(10).to_string(index=False))

    print(f"\n✅ Query intent extraction complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
