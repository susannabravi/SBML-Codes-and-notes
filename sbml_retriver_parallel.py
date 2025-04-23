import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from reactome2py import content

# === CONFIG ===
output_dir = './sbml_exports/'
input_file = './complete_list_of_pathways.txt'
retry_file = 'failed_downloads.txt'
max_workers = 15
retry_mode = False  # Set to True to retry from failed_downloads.txt

# === LOAD IDS ===
if retry_mode:
    with open(retry_file, 'r') as f:
        id_list = f.read().splitlines()
    print(f"Retrying {len(id_list)} previously failed downloads...")
else:
    with open(input_file, 'r') as file:
        pathways = file.read().splitlines()
    df = pd.DataFrame([line.split('\t') for line in pathways], columns=['ID', 'Pathway_Name', 'Species'])
    df_human = df[df['Species'] == 'Homo sapiens'].reset_index(drop=True)
    id_list = list(df_human['ID'])
    print(f"Starting download of {len(id_list)} Homo sapiens pathways")

# === SETUP ===
os.makedirs(output_dir, exist_ok=True)
failed_ids = []

# === DOWNLOAD FUNCTION ===
def download_pathway(path_id):
    # Check if file already exists, by matching start of filename
    existing_files = os.listdir(output_dir)
    if any(fname.startswith(path_id) for fname in existing_files):
        return f"Skipped {path_id} (already exists)"
    
    try:
        result = content.export_event(id=path_id, format='sbml', file=path_id, path=output_dir)
        return f"Exported {path_id}"
    except Exception as e:
        failed_ids.append(path_id)
        return f"Failed {path_id}: {str(e)}"


# === PARALLEL DOWNLOAD ===
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(download_pathway, pid) for pid in id_list]
    for future in as_completed(futures):
        print(future.result())

# === LOG FAILURES ===
if failed_ids:
    with open(retry_file, 'w') as f:
        for pid in failed_ids:
            f.write(f"{pid}\n")
    print(f"\nFailed to download {len(failed_ids)} pathways. Logged in '{retry_file}'.")
else:
    if retry_mode:
        print("\nAll previously failed pathways were downloaded successfully!")
    else:
        print("\nAll pathways downloaded successfully!")
