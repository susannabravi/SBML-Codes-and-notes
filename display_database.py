"""
Test script to display the dataset.
"""

import pandas as pd

# Set the maximum column width to None to ensure full content is displayed.
# pd.set_option('display.max_colwidth', None)

# Read the Parquet file into a DataFrame.
df = pd.read_parquet("./dataset_with_token_counts.parquet")

# Basic info
print("Dataset length:", len(df))
print("Columns in dataset:", df.columns.tolist())

# First rows 
print("\nFirst 5 rows of key columns:")
print(df.head())

# Display a sample of the dataset.
num_samples = 6  

# Iterate over the first num_samples rows and print full content without extra spaces.
print("\n--- Displaying full content for sample rows ---")
for idx in df.index[:num_samples]:
    print(f"\n--- Full content for sample row at index {idx} ---")
    print(f"Snippet:\n{df.at[idx, 'snippet']}")
    print(f"\nOriginal Notes:\n{df.at[idx, 'original_notes']}")
    print(f"\nCleaned Notes (without citations):\n{df.at[idx, 'notes']}")
    print(f"\nTokens (NLTK):\n{df.at[idx, 'notes_tokens_nltk']}")
    print(f"\nTokens Count (NLTK):\n{df.at[idx, 'notes_token_count_nltk']}")
    print(f"\nTokens (DEEP):\n{df.at[idx, 'notes_tokens_deep']}")
    print(f"\nTokens Count (DEEP):\n{df.at[idx, 'notes_token_count_deep']}")
    print(f"\nExtracted Citations:\n{df.at[idx, 'only_references']}")
    print("-" * 40)