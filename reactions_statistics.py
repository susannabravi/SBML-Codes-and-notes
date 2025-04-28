import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

# Set the style
plt.style.use('ggplot')

# Read the Parquet file into a DataFrame
df = pd.read_parquet("./reaction_counts.parquet")
reaction_count = df['reaction_count']

# Print of the stats
print("\nDescriptive Statistics for 'reaction_count':")
print(f"Count: {reaction_count.count()}")
print(f"Mean: {reaction_count.mean():.2f}")
print(f"Median: {reaction_count.median():.2f}")
print(f"Standard Deviation: {reaction_count.std():.2f}")
print(f"Minimum: {reaction_count.min()}")
print(f"Maximum: {reaction_count.max()}")
print(f"25th percentile (Q1): {np.percentile(reaction_count, 25):.2f}")
print(f"75th percentile (Q3): {np.percentile(reaction_count, 75):.2f}")
print(f"IQR (Interquartile Range): {np.percentile(reaction_count, 75) - np.percentile(reaction_count, 25):.2f}")

#
# Find the row with the maximum reaction count
max_idx = df['reaction_count'].idxmax()  # Get the index of the max reaction_count
max_row = df.loc[max_idx]                # Get the full row

print("\nFile with Maximum Reaction Count:")
print(f"file_id: {max_row['file_id']}")
print(f"reaction_count: {max_row['reaction_count']}")

# --- Plot distribution ---
plt.figure(figsize=(8, 4))
sns.histplot(reaction_count, kde=True, bins=30)
plt.title('Distribution of Reaction Counts')
plt.xlabel('Reaction Count')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.savefig('plot.png')

# --- without the max one ---
# Exclude the file with the maximum reaction count
df_filtered = df.drop(index=max_idx)

# Focus on 'reaction_count' column after exclusion
reaction_count = df_filtered['reaction_count']

# --- Descriptive statistics manually ---
print("\nDescriptive Statistics for 'reaction_count' (excluding max file):")
print(f"Count: {reaction_count.count()}")
print(f"Mean: {reaction_count.mean():.2f}")
print(f"Median: {reaction_count.median():.2f}")
print(f"Standard Deviation: {reaction_count.std():.2f}")
print(f"Minimum: {reaction_count.min()}")
print(f"Maximum: {reaction_count.max()}")
print(f"25th percentile (Q1): {np.percentile(reaction_count, 25):.2f}")
print(f"75th percentile (Q3): {np.percentile(reaction_count, 75):.2f}")
print(f"IQR (Interquartile Range): {np.percentile(reaction_count, 75) - np.percentile(reaction_count, 25):.2f}")

# --- Plot distribution ---
plt.figure(figsize=(8, 4))
sns.histplot(reaction_count, kde=True, bins=30)
plt.title('Distribution of Reaction Counts (excluding max file)')
plt.xlabel('Reaction Count')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()

plt.savefig('plot2.png')

# Sort the DataFrame by reaction_count in descending order
sorted_df = df.sort_values(by='reaction_count', ascending=False)

# Print the top files
print("\nFiles sorted by reaction_count (descending):")
print(sorted_df.head(30))
