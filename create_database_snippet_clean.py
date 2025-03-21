import os
import xml.etree.ElementTree as ET
import pyarrow as pa
import pyarrow.parquet as pq

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
    # Create a new element with the local tag name.
    new_elem = ET.Element(tag)
    
    # Copy attributes without namespace prefixes.
    for attr, value in elem.attrib.items():
        local_attr = attr.split('}')[-1] if '}' in attr else attr
        new_elem.set(local_attr, value)
    
    # Copy text and tail.
    new_elem.text = elem.text
    new_elem.tail = elem.tail
    
    # Recursively process child elements.
    for child in elem:
        new_child = remove_namespaces_and_elements(child)
        if new_child is not None:
            new_elem.append(new_child)

    return new_elem

# Specify the directory containing the SBML files.
data_dir = "/Users/susannabravi/Documents/DS/Tesi/ExtractionProva/sbml_exports"  # Modify as needed.
records = []  # This list will store the extracted reaction records.

# Iterate over all SBML files in the directory.
for filename in os.listdir(data_dir):
    if not filename.endswith('.sbml'):
        continue  # Skip files that do not have a .sbml extension.
    file_path = os.path.join(data_dir, filename)
    
    # Parse the SBML file.
    try:
        tree = ET.parse(file_path)
    except ET.ParseError as e:
        print(f"Error parsing {filename}: {e}.")
        continue
    root = tree.getroot()

    # Determine the default namespace from the root tag.
    # The tag looks like '{namespace}tagname' if a default namespace is declared.
    default_ns = {}
    reaction_xpath = ".//reaction"
    if root.tag.startswith("{"):
        ns_uri = root.tag[1:root.tag.index("}")]
        default_ns['sbml'] = ns_uri  # Map the 'sbml' prefix to the default namespace.
        reaction_xpath = ".//sbml:reaction"  # Adjust XPath to include the namespace.
    
    # Find all reaction elements using the appropriate XPath.
    reactions = root.findall(reaction_xpath, default_ns)
    print(f"Found {len(reactions)} reaction(s) in {filename}.")
    
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
        
        # Append the extracted data to the records list, including the file identifier.
        records.append({
            'file_id': filename,  # The file identifier is set to the filename.
            'reaction_id': reaction_id,
            'notes': notes_text,
            'snippet': snippet
        })

# Create a PyArrow table from the extracted records.
table = pa.Table.from_pydict({
    'file_id': [record['file_id'] for record in records],
    'reaction_id': [record['reaction_id'] for record in records],
    'notes': [record['notes'] for record in records],
    'snippet': [record['snippet'] for record in records]
})

# Save the table to a Parquet file.
pq.write_table(table, 'sbml_exports/reactions.parquet')
print("Parquet file 'reactions.parquet' has been created.")
