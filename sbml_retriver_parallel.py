import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from reactome2py import content

# Costants
output_dir = './sbml_exports/'
input_file = './complete_list_of_pathways_nuova.txt'
retry_file = 'failed_downloads.txt'
max_workers = min(int(os.cpu_count()/2), 15)
retry_mode = False  # Set to True to retry from failed_downloads.txt


# Open the list of pathways or the failed_download file.
if retry_mode:
    with open(retry_file, 'r') as f:
        id_list = f.read().splitlines()
    print(f"Retrying {len(id_list)} previously failed downloads")
else:
    with open(input_file, 'r') as file:
        pathways = file.read().splitlines()
    df = pd.DataFrame([line.split('\t') for line in pathways], columns=['ID', 'Pathway_Name', 'Species'])
    df_human = df[df['Species'] == 'Homo sapiens'].reset_index(drop=True)
    id_list = list(df_human['ID'])
    print(f"Starting download of {len(id_list)} Homo sapiens pathways")

# Create output directory if doesn't exist.
os.makedirs(output_dir, exist_ok=True)
failed_ids = []

# Dowload the files from reactome
def download_pathway(path_id):
    # Check if file already exists, by matching start of filename
    # If the file already exists, skip it.
    existing_files = os.listdir(output_dir)
    if any(fname.startswith(path_id) for fname in existing_files):
        print(f"Skipped {path_id} (already exists)")
    try:
        result = content.export_event(id=path_id, format='sbml', file=path_id, path=output_dir)
    except Exception as e:
        failed_ids.append(path_id)
        return f"Failed {path_id}: {str(e)}"


# Parallel download 
with ThreadPoolExecutor(max_workers = max_workers) as executor:
    futures = [executor.submit(download_pathway, pid) for pid in id_list]

# Save faildes_ids to retry file.
if failed_ids:
    with open(retry_file, 'w') as f:
        f.write("\n".join(failed_ids))
    print(f"Failed downloads saved to {retry_file}")
else:
    print("No failed downloads.")