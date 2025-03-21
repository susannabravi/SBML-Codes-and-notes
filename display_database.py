"""
Test script to display the dataset.
"""

import pandas as pd

# Set the maximum column width to None to ensure full content is displayed.
# pd.set_option('display.max_colwidth', None)

# Read the Parquet file into a DataFrame.
df = pd.read_parquet("/Users/susannabravi/Documents/DS/Tesi/ExtractionProva/sbml_exports/reactions.parquet")
print("dataset length",len(df))
print(df.head())
# Display the complete first row.
print(df['snippet'].iloc[0])