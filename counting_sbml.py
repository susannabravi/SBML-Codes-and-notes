# Test script:
#  1) see if the number of files in the sbml folder is the same as in the list of pathways
#  2) look for the exact match between the sbml id in the folder and in the list file

import os
import pandas as pd
folder_path = './sbml_exports/' 

# 1)
# Counting files in the sbml_export folder.
file_count = len([name for name in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, name))
])

print(f"Total files in '{folder_path}': {file_count}")

# Counting rows of human pathways in the list file.
with open('./complete_list_of_pathways.txt', 'r') as file:
    pathways = file.read().splitlines()
df = pd.DataFrame([line.split('\t') for line in pathways], columns=['ID', 'Pathway_Name', 'Species'])
df_human = df[df['Species'] == 'Homo sapiens'].reset_index(drop=True)
print(f"Total human pathways: {len(df_human)}")

# 2)
# list all the files in 'sbml_export'
existing_files = os.listdir(folder_path)
# list of all the path id
id_list = list(df_human['ID'])
counter = 0
# check
for pid in id_list:
    if any(fname.startswith(pid) for fname in existing_files):
        counter += 1

if counter == len(df_human):
    print(f"All the {counter} files are present in the SBML folder")
else:
    print(f"Not all the files are present in the SBML folder")