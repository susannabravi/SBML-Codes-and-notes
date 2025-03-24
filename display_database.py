"""
Test script to display the dataset.
"""

import pandas as pd

# Set the maximum column width to None to ensure full content is displayed.
# pd.set_option('display.max_colwidth', None)

# Read the Parquet file into a DataFrame.
df = pd.read_parquet("./train.parquet")
print("dataset length",len(df))
print(df.head())
# Display the complete first row.
print(df['snippet'].iloc[0])