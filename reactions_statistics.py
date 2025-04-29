import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

# Set plots style
plt.style.use('ggplot')

# Read the Parquet file into a DataFrame
df = pd.read_parquet("./reaction_counts.parquet")
reaction_count = df['reaction_count']

# Function to plot data
def plot_data(data, logscale_x=False, logscale_y=False, outputname="fig1.png", n_bins=30, boxplot=False):
    if logscale_x:
        data = np.log(data)

    if boxplot:
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, axs = plt.subplots(1, 1, figsize=(8, 4))
        axs = [axs]  

    sns.histplot(data, kde=True, bins=n_bins, ax=axs[0])
    axs[0].set_title('Distribution of Reaction Counts')
    axs[0].set_xlabel('Reaction Count' if not logscale_x else 'log(Reaction Count)')
    axs[0].set_ylabel('Frequency')
    axs[0].grid(True)

    if logscale_y:
        axs[0].set_yscale('log')

    if boxplot:
        sns.boxplot(data, ax=axs[1], orient='h')
        axs[1].set_xlabel('Reaction Count' if not logscale_x else 'log(Reaction Count)')
        axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(outputname)
    plt.close()

# Function to print stats
def print_stats(data):
    print(f"\nDescriptive Statistics:")
    print(f"Count: {data.count()}")
    print(f"Mean: {data.mean():.2f}")
    print(f"Median: {data.median():.2f}")
    print(f"Standard Deviation: {data.std():.2f}")
    print(f"Minimum: {data.min()}")
    print(f"Maximum: {data.max()}")
    print(f"25th percentile (Q1): {np.percentile(data, 25):.2f}")
    print(f"75th percentile (Q3): {np.percentile(data, 75):.2f}")
    print(f"IQR (Interquartile Range): {np.percentile(data, 75) - np.percentile(data, 25):.2f}")

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

# Filtered data 95th percentile -------------------
perc = np.percentile(df['reaction_count'], 90)
df_filtered = df[df['reaction_count'] <= perc]
reaction_count_filtered = df_filtered['reaction_count']

plot_data(reaction_count_filtered, boxplot = True, outputname = "fig3.png")
print_stats(reaction_count_filtered)

