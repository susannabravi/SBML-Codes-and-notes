import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
from descriptive_stat_functions import plot_data, print_stats

df = pd.read_parquet("./reactions.parquet")
ggplot_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# No references data -----------------------
df_nocit = df[df['only_references'].isna()]

# Basic info
print("Dataset length:", len(df_nocit))
print("Columns in dataset:", df_nocit.columns.tolist())

# First rows 
print("\nFirst 5 rows:")
print(df_nocit.head(5))

# Display a sample of the dataset.
num_samples = 2 

# Iterate over the first num_samples rows and print full content without extra spaces.
print("\n--- Displaying full content for sample rows ---")
for idx in df_nocit.index[:num_samples]:
    print(f"\n--- Full content for sample row at index {idx} ---")
    print(f"Snippet:\n{df_nocit.at[idx, 'snippet']}")
    print(f"\nOriginal Notes:\n{df_nocit.at[idx, 'original_notes']}")
    print(f"\nCleaned Notes (without citations):\n{df_nocit.at[idx, 'notes']}")
    print(f"\nExtracted Citations:\n{df_nocit.at[idx, 'only_references']}")
    print("-" * 40)

#for idx in df_nocit.index[:100]:
#    print(f"\nOriginal Notes:\n{df_nocit.at[idx, 'original_notes']}")

# References per file statistics --------------------
# Example: 
'''(Duncan et al. 2002, Sundheim et al. 2006, Chen et al. 2010, Dango et al. 2011);
 (Duncan et al. 2002, Sundheim et al. 2006)'''
# Function to clean the references ---> should I add it to the main dataset? 
def clean_references(entry):
    if not isinstance(entry, str):
        return []
    refs = []
    # Split by semicolon because each row have a list separated by ;
    for group in entry.split(';'):
        # Strip white spaces and brackets
        group = group.strip().strip('()')
        # Split by comma
        refs.extend([ref.strip() for ref in group.split(',')])
    return refs

df['parsed_references'] = df['only_references'].apply(clean_references)

# How many unique references in general
# How many references per file 
# List of all ref
refs_list_per_file = df.groupby('file_id')['parsed_references'].apply(
    lambda lists: [ref for refs in lists for ref in refs]
)
refs_per_file = refs_list_per_file.apply(lambda refs: len(refs))
# How many unique references per file 
unique_refs_per_file = refs_list_per_file.apply(lambda refs: len(set(refs)))
df_refs_per_file = pd.DataFrame({
    'file_id': refs_list_per_file.index,
    'all_references': refs_list_per_file.values,
    'total_references': refs_per_file.values,
    'unique_references': unique_refs_per_file.values
})
print(df_refs_per_file.head(10))
print(f"\nTotal number of references in all files (with duplicates): {sum(df_refs_per_file.total_references.values)}")
print(f"\nTotal number of unique references across all files: {sum(df_refs_per_file.unique_references.values)}")

plot_data(data = df_refs_per_file['total_references'],
          outputname = "fig4.png",
          title = "References per file (with duplicates)",
          xlabel = "References",
          color = ggplot_colors[1],
          boxplot = True
          )
plot_data(data =  df_refs_per_file['total_references'],
          outputname = "fig5.png",
          title = "Referencese per reactions with log scale x",
          xlabel = "log(References)",
          color = ggplot_colors[1],
          boxplot = True,
          logscale_x = True
          )
plot_data(data =  df_refs_per_file['total_references'],
          outputname = "fig6.png",
          title = "Referencese per reactions cutted",
          xlabel = "References",
          color = ggplot_colors[1],
          boxplot = True,
          cut_percentile=(0,90)
          )

print_stats(df_refs_per_file['total_references'])

# References per reactions statistics -----------------------------
# INSPECT REACTIONS
# While pathways (file_id) are all unique, the reactions inside them are not, so if I group for reaction_id
# I get less reactions than the total length of the dataframe.
df2 = df.drop(["file_id","parsed_references"], axis = 1)
# Analyze the unique values in each column
print(f"\nUnique values for each column:\n{df.drop('parsed_references',axis=1).nunique()}")
# Drop column file_id in order to remove the duplicates.
print(f"\nNumber of rows in the original data:{len(df)}")
df2 = df2.drop_duplicates().reset_index(drop=True)
print(f"\nNumber of rows after dropping duplicates:{len(df2)}")
# Sort the data and display the first rows
print(f"\nFirst rows of sorted dataset\n{df2.sort_values(by='reaction_id').head(10)}")

#Total number of reactions: 15357

# How many references per reaction
references_per_reaction = df.groupby('reaction_id')['parsed_references'].apply(
    lambda lists: sum(len(ref) for ref in lists)
)
# How many unique references per file 
unique_refs_per_reactions = df.groupby('reaction_id')['parsed_references'].apply(
    lambda ref_lists: len(set(ref for refs in ref_lists for ref in refs))
)

# Results
print(f"\nTotal number of references per reaction:\n{references_per_reaction.head(10)}") 
print(f"\nUnique references per reaction:\n{unique_refs_per_reactions.head(10)}") 

plot_data(data = references_per_reaction,
          outputname = "fig7.png",
          title = "Referencese per reactions",
          xlabel = "References",
          color = ggplot_colors[2],
          boxplot = True
          )
plot_data(data = references_per_reaction,
          outputname = "fig8.png",
          title = "Referencese per reactions",
          xlabel = "References",
          color = ggplot_colors[2],
          boxplot = True,
          logscale_x = True
          )
plot_data(data = references_per_reaction,
          outputname = "fig9.png",
          title = "Referencese per reactions",
          xlabel = "References",
          color = ggplot_colors[2],
          boxplot = True,
          cut_percentile=(0,90)
          )

print_stats(references_per_reaction)