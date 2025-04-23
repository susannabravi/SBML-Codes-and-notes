import os
import pandas as pd
folder_path = './sbml_exports/'  # or whatever path you're using

file_count = len([
    name for name in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, name))
])

print(f"Total files in '{folder_path}': {file_count}")

with open('./complete_list_of_pathways_nuova.txt', 'r') as file:
    pathways = file.read().splitlines()
df = pd.DataFrame([line.split('\t') for line in pathways], columns=['ID', 'Pathway_Name', 'Species'])
df_human = df[df['Species'] == 'Homo sapiens'].reset_index(drop=True)
print(len(df_human))
