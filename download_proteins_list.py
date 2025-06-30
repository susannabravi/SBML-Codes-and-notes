import pandas as pd
import reactome2py.content as reactome
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

max_workers = min(int(os.cpu_count()/2), 15)

def process_pathway(pathway_id):
    proteins = set()
    try:
        participants = reactome.participants_physical_entities(pathway_id)
        
        for participant in participants:
            if participant.get('schemaClass') == 'EntityWithAccessionedSequence':
                name = participant.get('displayName', '')
                if name:
                    clean_name = name.split(' [')[0].strip()
                    
                    if (len(clean_name) > 2 and not clean_name.isdigit() and 
                        not clean_name.endswith(' gene') and 
                        not clean_name.endswith(' activated')):
                        proteins.add(clean_name)
    except:
        pass
    
    return proteins

def extract_proteins(input_file='complete_list_of_pathways_nuova.txt'):
    # Load pathways
    print("Loading pathway file...")
    with open(input_file, 'r') as file:
        pathways = file.read().splitlines()
    
    df = pd.DataFrame([line.split('\t') for line in pathways], columns=['ID', 'Pathway_Name', 'Species'])
    df_human = df[df['Species'] == 'Homo sapiens'].reset_index(drop=True)
    pathway_ids = list(df_human['ID'])
    
    max_pathways = len(pathway_ids)
    print(f"Processing {max_pathways} human pathways in parallel...")
    
    all_proteins = set()
    completed = 0
    
    # Process pathways in parallel with progress tracking
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pathway = {executor.submit(process_pathway, pathway_id): pathway_id 
                            for pathway_id in pathway_ids}
        
        for future in as_completed(future_to_pathway):
            completed += 1
            if completed % 50 == 0 or completed == max_pathways:
                print(f"Progress: {completed}/{max_pathways}")
            
            protein_set = future.result()
            all_proteins.update(protein_set)
    
    print(f"\nFound {len(all_proteins)} unique proteins")
    
    if all_proteins:
        # Save results
        with open('reactome_proteins_clean.txt', 'w') as f:
            for protein in sorted(all_proteins):
                f.write(protein + '\n')
        
        print("Saved to reactome_proteins_clean.txt")
    return list(all_proteins)

extract_proteins()
print("Finishhh")