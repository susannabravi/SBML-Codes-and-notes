import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
from descriptive_stat_functions import plot_data, print_stats

# Read the Parquet file into a DataFrame
df = pd.read_parquet("./reaction_counts.parquet")
reaction_count = df['reaction_count']

# Raw data ----------------------------------------
plot_data(reaction_count, boxplot = True)
print_stats(reaction_count)

# Find the row with the maximum reaction count
max_idx = df['reaction_count'].idxmax()  
max_row = df.loc[max_idx]   

print("\nFile with Maximum Reaction Count:")
print(f"file_id: {max_row['file_id']}")
print(f"reaction_count: {max_row['reaction_count']}")
# R-HSA-162582.sbml

# Sort the DataFrame by reaction_count in descending order
sorted_df = df.sort_values(by='reaction_count', ascending=False)

# Print the top files
print("\nFiles sorted by reaction_count (descending):")
print(sorted_df.head(15))

# Log-scaled data ---------------------------------
plot_data(reaction_count, logscale_x = True, outputname = "fig2.png")

# Filtered data 90th percentile -------------------
perc = np.percentile(df['reaction_count'], 90)
df_filtered = df[df['reaction_count'] <= perc]
reaction_count_filtered = df_filtered['reaction_count']

plot_data(reaction_count_filtered, boxplot = True, outputname = "fig3.png")
print_stats(reaction_count_filtered)

# Non sarebbe male vedere che c'è dal 90esimo percentile in poi... 

