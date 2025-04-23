import os
import xml.etree.ElementTree as ET
import pyarrow as pa
import pyarrow.parquet as pq
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

ns = {
    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    'dc': 'http://purl.org/dc/elements/1.1/',
    'vCard': 'http://www.w3.org/2001/vcard-rdf/3.0#',
    'dcterms': 'http://purl.org/dc/terms/',
    'bqbiol': 'http://biomodels.net/biology-qualifiers/',
    'html': 'http://www.w3.org/1999/xhtml'
}

def remove_namespaces_and_elements(elem):
    tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
    if tag in ('notes', 'annotation'):
        return None
    new_elem = ET.Element(tag)
    for attr, value in elem.attrib.items():
        local_attr = attr.split('}')[-1] if '}' in attr else attr
        new_elem.set(local_attr, value)
    new_elem.text = elem.text
    new_elem.tail = elem.tail
    for child in elem:
        new_child = remove_namespaces_and_elements(child)
        if new_child is not None:
            new_elem.append(new_child)
    return new_elem

def remove_citations(text):
    if not text:
        return text, None
    pattern = r'\(\s*[A-Z][^)]*et al\.[^)]*\s?\d{4}\)'
    citations = re.findall(pattern, text)
    cleaned_text = re.sub(pattern, '', text)
    cleaned_text = re.sub(r'\s+([.,;:!?])', r'\1', cleaned_text)
    cleaned_text = " ".join(cleaned_text.split())
    citations_str = "; ".join(citations) if citations else None
    return cleaned_text, citations_str

def process_file(file_path):
    filename = os.path.basename(file_path)
    records = []
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing {filename}: {e}")
        return records

    default_ns = {}
    reaction_xpath = ".//reaction"
    if root.tag.startswith("{"):
        ns_uri = root.tag[1:root.tag.index("}")]
        default_ns['sbml'] = ns_uri
        reaction_xpath = ".//sbml:reaction"

    reactions = root.findall(reaction_xpath, default_ns)
    print(f"Found {len(reactions)} reaction(s) in {filename}.")

    for reaction in reactions:
        reaction_id = reaction.get('id')
        clean_reaction = remove_namespaces_and_elements(reaction)
        snippet = ET.tostring(clean_reaction, encoding='unicode')

        notes_elem = reaction.find('sbml:notes', default_ns) if default_ns else reaction.find('notes')
        notes_text = None
        if notes_elem is not None:
            p_elem = notes_elem.find('html:p', ns)
            if p_elem is not None and p_elem.text:
                notes_text = p_elem.text.strip()
            elif notes_elem.text:
                notes_text = notes_elem.text.strip()

        original_notes = notes_text
        cleaned_notes, only_references = remove_citations(notes_text) if notes_text else (None, None)

        records.append({
            'file_id': filename,
            'reaction_id': reaction_id,
            'original_notes': original_notes,
            'notes': cleaned_notes,
            'only_references': only_references,
            'snippet': snippet
        })

    return records

def main():
    data_dir = "./sbml_exports"
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".sbml")]
    total_files = len(files)
    all_records = []
    completed = 0

    print(f"Processing {total_files} SBML files...")

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_file, f): f for f in files}
        for future in as_completed(futures):
            file_path = futures[future]
            result = future.result()
            all_records.extend(result)
            completed += 1
            print(f"[{completed}/{total_files}] Processed: {os.path.basename(file_path)}")

    print(f"\nFinished processing {completed} of {total_files} files.")

    table = pa.Table.from_pydict({
        'file_id': [r['file_id'] for r in all_records],
        'reaction_id': [r['reaction_id'] for r in all_records],
        'original_notes': [r['original_notes'] for r in all_records],
        'notes': [r['notes'] for r in all_records],
        'only_references': [r['only_references'] for r in all_records],
        'snippet': [r['snippet'] for r in all_records]
    })
    pq.write_table(table, './reactions.parquet')
    print("Parquet file 'reactions.parquet' has been created.")

if __name__ == "__main__":
    main()
