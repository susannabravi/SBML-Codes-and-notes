"""
Test script to display the dataset.
"""

import pandas as pd

# Set the maximum column width to None to ensure full content is displayed.
# pd.set_option('display.max_colwidth', None)

# Read the Parquet file into a DataFrame.
df = pd.read_parquet("./reactions.parquet")

# Basic info
print("Dataset length:", len(df))
print("Columns in dataset:", df.columns.tolist())

# First rows 
print("\nFirst 5 rows of key columns:")
print(df.head())

# Display a sample of the dataset.
num_samples = 5  # You can adjust this number as needed.

# Iterate over the first num_samples rows and print full content.
print("\n--- Displaying full content for sample rows ---")
for idx in df.index[:num_samples]:
    print(f"\n--- Full content for sample row at index {idx} ---")
    print("Snippet:\n", df.at[idx, 'snippet'])
    print("\nOriginal Notes:\n", df.at[idx, 'original_notes'])
    print("\nCleaned Notes (without citations):\n", df.at[idx, 'notes'])
    print("\nExtracted Citations:\n", df.at[idx, 'only_references'])
    print("-" * 40)