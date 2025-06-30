"""
Test script to display the dataset.
"""

import pandas as pd
import numpy as np

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
   # print(f"\nOriginal Notes:\n{df.at[idx, 'original_notes']}")
    print(f"\nCleaned Notes (without citations):\n{df.at[idx, 'notes']}")
    print(f"\nTokens (NLTK):\n{df.at[idx, 'notes_tokens_nltk']}")
    print(f"\nTokens Count (NLTK):\n{df.at[idx, 'notes_token_count_nltk']}")
    print(f"\nTokens (DEEP):\n{df.at[idx, 'notes_tokens_deep']}")
    print(f"\nTokens Count (DEEP):\n{df.at[idx, 'notes_token_count_deep']}")
    print(f"\nTokens (DEEP_EXTENDED):\n{df.at[idx, 'notes_tokens_extended']}")
    print(f"\nTokens Count (DEEP_EXTENDED):\n{df.at[idx, 'notes_token_count_extended']}")
   # print(f"\nExtracted Citations:\n{df.at[idx, 'only_references']}")
    print("-" * 40)

def print_stats(data):
    print(f"\nDescriptive Statistics:")
    #print(f"Count: {data.count()}")
    print(f"Mean: {data.mean():.2f}")
    print(f"Median: {data.median():.2f}")
    print(f"Standard Deviation: {data.std():.2f}")
    print(f"Minimum: {data.min()}")
    print(f"Maximum: {data.max()}")
    print(f"5th percentile: {np.percentile(data, 5):.2f}")
    print(f"25th percentile (Q1): {np.percentile(data, 25):.2f}")
    print(f"75th percentile (Q3): {np.percentile(data, 75):.2f}")
    print(f"95th percentile: {np.percentile(data, 95):.2f}")
    print(f"IQR (Interquartile Range): {np.percentile(data, 75) - np.percentile(data, 25):.2f}")

print("-" * 40)
print("NLTK tokenizer")
print_stats(df["notes_token_count_nltk"])
print("-" * 40)
print("DeepSeek tokenizer")
print_stats(df["notes_token_count_deep"])
print("-" * 40)
print("DeepSeek tokenizer")
print_stats(df["notes_token_count_extended"])
