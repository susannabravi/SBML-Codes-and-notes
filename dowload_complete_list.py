import requests

# Link to reactomr and output file 
url = "https://reactome.org/download/current/ReactomePathways.txt"
output_file = "complete_list_of_pathways_nuova.txt"

print(f"Downloading pathway list from Reactome")
response = requests.get(url)

if response.status_code == 200:
    with open(output_file, 'wb') as f:
        f.write(response.content)
    print(f"Saved to '{output_file}'")
else:
    print(f"Failed to download. Status code: {response.status_code}")
