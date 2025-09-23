import pandas as pd
from descriptive_stat_functions import extract_references
import pyarrow.parquet as pq
'''
Create the pathway data with colunms:
File_id | Reaction_count | Total References | Unique References | All References | Notes Count
'''
df_path = pd.read_parquet("../pathways.parquet")
df = pd.read_parquet("../reactions.parquet")

df['parsed_references'] = df['only_references'].apply(extract_references)


refs_list_per_file = df.groupby('file_id')['parsed_references'].apply(
    lambda lists: [ref for refs in lists for ref in refs]
)
refs_per_file = refs_list_per_file.apply(lambda refs: len(refs))
# How many unique references per file 
unique_refs_per_file = refs_list_per_file.apply(lambda refs: len(set(refs)))
notes_count = df.groupby('file_id')['notes'].count().values
df_refs_per_file = pd.DataFrame({
    'file_id': refs_list_per_file.index,
    'notes_count': notes_count,
    'all_references': refs_list_per_file.values,
    'total_references': refs_per_file.values,
    'unique_references': unique_refs_per_file.values
})

df_merged = pd.merge(df_path, df_refs_per_file, on='file_id')

df_merged.to_parquet('pathways.parquet', engine='pyarrow')
print("Parquet file 'pathways.parquet' has been created.")