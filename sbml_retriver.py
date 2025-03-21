import os
import pandas as pd
from reactome2py import content

# Read and process the text file with pathway details
with open('/Users/susannabravi/Documents/DS/Tesi/ExtractionProva/complete_list_of_pathways.txt', 'r') as file:
    pathways = file.read().splitlines()
df = pd.DataFrame([line.split('\t') for line in pathways], columns=['ID', 'Pathway_Name', 'Species'])
df_human = df[df['Species'] == 'Homo sapiens'].reset_index(drop=True)
id_list = list(df_human['ID'])

# Try with only the first 20 files 
id_list = id_list[:100] 

# Define the output directory for SBML files
output_dir = './sbml_exports/'
os.makedirs(output_dir, exist_ok=True)

# Loop through each pathway ID and export the SBML
for path_id in id_list:
    filename = f'{path_id}'
    result = content.export_event(id=path_id, format='sbml', file=filename, path=output_dir)
    print(f"Exported pathway {path_id} to {os.path.join(output_dir, filename)}")
