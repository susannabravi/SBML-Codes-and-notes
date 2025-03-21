# This script is taken from Riccardo De Luca https://github.com/rickydeluca/biopath-generator

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

def remove_namespace(element):
    # Remove namespace from this element's tag if present.
    if '}' in element.tag:
        element.tag = element.tag.split('}', 1)[1]
    # Process child elements recursively.
    for child in element:
        remove_namespace(child)


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

        remove_namespace(reaction)
        # Extract the reaction's id.
        reaction_id = reaction.get('id')
        
        # Convert the reaction element to a string (the full snippet).
        snippet = ET.tostring(reaction, encoding='unicode')
        
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
        '''
        # Extract PubMed IDs from the annotation section.
        pubmed_ids = []
        # The annotation element is also likely in the SBML namespace.
        annotation = reaction.find('sbml:annotation', default_ns) if default_ns else reaction.find('annotation')
        if annotation is not None:
            rdf = annotation.find('rdf:RDF', ns)
            if rdf is not None:
                # Iterate over each RDF description element.
                for description in rdf.findall('rdf:Description', ns):
                    # Look for the 'isDescribedBy' element that contains PubMed references.
                    for isDescribedBy in description.findall('bqbiol:isDescribedBy', ns):
                        bag = isDescribedBy.find('rdf:Bag', ns)
                        if bag is not None:
                            # Each <rdf:li> element in the bag holds a reference.
                            for li in bag.findall('rdf:li', ns):
                                resource = li.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource')
                                if resource and 'pubmed' in resource:
                                    pubmed_ids.append(resource)'
        '''
        # Append the extracted data to the records list, including the file identifier.
        records.append({
            'file_id': filename,  # The file identifier is set to the filename.
            'reaction_id': reaction_id,
            'notes': notes_text,
        #   'pubmed_ids': pubmed_ids,
            'snippet': snippet
        })

# Create a PyArrow table from the extracted records.
table = pa.Table.from_pydict({
    'file_id': [record['file_id'] for record in records],
    'reaction_id': [record['reaction_id'] for record in records],
    'notes': [record['notes'] for record in records],
#   'pubmed_ids': [record['pubmed_ids'] for record in records],
    'snippet': [record['snippet'] for record in records]
})

# Save the table to a Parquet file.
pq.write_table(table, 'sbml_exports/reactions.parquet')
print("Parquet file 'reactions.parquet' has been created.")