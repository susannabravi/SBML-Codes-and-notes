import os
import xml.etree.ElementTree as ET
import pyarrow as pa
import pyarrow.parquet as pq
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

# Define num of max_workers.
max_workers = int(os.cpu_count()/2)

# Define namespaces for RDF and XHTML.
ns = {
    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    'dc': 'http://purl.org/dc/elements/1.1/',
    'vCard': 'http://www.w3.org/2001/vcard-rdf/3.0#',
    'dcterms': 'http://purl.org/dc/terms/',
    'bqbiol': 'http://biomodels.net/biology-qualifiers/',
    'html': 'http://www.w3.org/1999/xhtml'
}

# Define a function to remove namespaces and skip 'notes' and 'annotation' elements for the snippet part.
def remove_namespaces_and_elements(elem):
    # Get local tag name (removing namespace if present)
    tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
    # Skip the element if it is 'notes' or 'annotation'
    if tag in ('notes', 'annotation'):
        return None
    # Create a new element with the local tag name (recursive way).
    new_elem = ET.Element(tag)

    # Copy attributes without namespace prefixes.
    for attr, value in elem.attrib.items():
        local_attr = attr.split('}')[-1] if '}' in attr else attr
        if local_attr == 'metaid':
            continue 
        new_elem.set(local_attr, value)

    # Copy text and tail.
    new_elem.text = elem.text
    new_elem.tail = elem.tail

    # Recursively process child elements
    for child in elem:
        new_child = remove_namespaces_and_elements(child)
        if new_child is not None:
            new_elem.append(new_child)
    return new_elem

def remove_citations(text):
    if not text:
        return text, None

    citations = []

    # Pattern 1: (Amin et al, 2016), (Lambert 2008))
    paren_pattern = r'\(\s*(?:e\.g\.\s*)?[A-Z][^()]*?\d{4}[a-z]?\s*\)'
    paren_citations = re.findall(paren_pattern, text)
    citations.extend(paren_citations)
    text = re.sub(paren_pattern, '', text)

    # Pattern 2: (Amin et al, 2016 or Katayama et al 2014)
    inline_pattern = r'\b[A-Z][a-z]+(?:\s(?:et al\.?|and\s[A-Z][a-z]+)?)?(?:,)?\s+et al\.?(?:,)?\s*\d{4}[a-z]?\b'
    inline_citations = re.findall(inline_pattern, text)
    citations.extend(inline_citations)
    text = re.sub(inline_pattern, '', text)

    # Clean up 
    text = re.sub(r'\s*:\s*', ': ', text)  # fix spacing after colons
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # remove space before punctuation
    text = " ".join(text.split())  # remove excessive whitespace

    citations_str = "; ".join(citations) if citations else None

    return text, citations_str



def process_file(file_path):
    filename = os.path.basename(file_path)
    records = []
    reaction_count = 0

    # Parse the SBML file.
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing {filename}: {e}")
        return records

    # Determine the default namespace from the root tag.
    # The tag looks like '{namespace}tagname' if a default namespace is declared.
    default_ns = {}
    reaction_xpath = ".//reaction"
    if root.tag.startswith("{"):
        ns_uri = root.tag[1:root.tag.index("}")]
        default_ns['sbml'] = ns_uri
        reaction_xpath = ".//sbml:reaction"

    reactions = root.findall(reaction_xpath, default_ns)
    reaction_count = len(reactions)
    print(f"Found {reaction_count} reaction(s) in {filename}.")

    # Iterate over each reaction element.
    for reaction in reactions:
        # Extract the reaction's id.
        reaction_id = reaction.get('id')
        
        # Modified snippet: remove namespaces and delete notes and annotation elements.
        clean_reaction = remove_namespaces_and_elements(reaction)
        snippet = ET.tostring(clean_reaction, encoding='unicode')
        
        # Extract the natural language description from the notes section.
        notes_text = None
        # The notes element is likely in the SBML namespace.
        notes_elem = reaction.find('sbml:notes', default_ns) if default_ns else reaction.find('notes')
        if notes_elem is not None:
            # Look for a <p> element within the notes using the XHTML namespace.
            p_elem = notes_elem.find('html:p', ns)
            if p_elem is not None and p_elem.text:
                notes_text = p_elem.text.strip()
            elif notes_elem.text:
                notes_text = notes_elem.text.strip()
         # Store the original notes before processing.
        original_notes = notes_text
        
        # Remove citations from the notes.
        if notes_text:
            cleaned_notes, only_references = remove_citations(notes_text)
        else:
            cleaned_notes, only_references = notes_text, None
        
        # Append the extracted data including original and processed notes.
        records.append({
            'file_id': filename,
            'reaction_id': reaction_id,
            'original_notes': original_notes,
            'notes': cleaned_notes,
            'only_references': only_references,
            'snippet': snippet
        })

    return records, reaction_count

def main():
    data_dir = "./sbml_exports"
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".sbml")]
    total_files = len(files)
    all_records = []
    reaction_counts = []
    completed = 0

    print(f"Processing {total_files} SBML files...")

    # Parallel part 
    with ProcessPoolExecutor(max_workers = max_workers) as executor:
        futures = {executor.submit(process_file, f): f for f in files}
        for future in as_completed(futures):
            file_path = futures[future]
            result,reaction_count = future.result()
            all_records.extend(result)
            reaction_counts.append({'file_id': os.path.basename(file_path), 'reaction_count': reaction_count})
            completed += 1
            print(f"[{completed}/{total_files}] Processed: {os.path.basename(file_path)}")

    print(f"\nFinished processing {completed} of {total_files} files.")

    # Create table for dataset
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

    # Create table for reaction counts
    reaction_count_table = pa.Table.from_pydict({
        'file_id': [r['file_id'] for r in reaction_counts],
        'reaction_count': [r['reaction_count'] for r in reaction_counts]
    })
    pq.write_table(reaction_count_table, './reaction_counts.parquet')
    print("Parquet file 'reactions_per_file.parquet' has been created.")

# Necessary for using ProcessPoolExecutor
if __name__ == "__main__":
    main()
