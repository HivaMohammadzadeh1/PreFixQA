import os
import json
from openai import AzureOpenAI
from llama_parse import LlamaParse
from copy import deepcopy
from llama_index.core.schema import TextNode
import nest_asyncio; nest_asyncio.apply()

### Setup the OpenAI or Azure Key 
#client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "YOUR_KEY_HERE"))

client = AzureOpenAI(
  azure_endpoint = "YOUR_AZURE_ENDPOINT",
  api_key=os.getenv("AZURE_OPENAI_KEY", "YOUR_KEY_HERE"),
  api_version="YOUR_API_VERSION"
)

### Setup LlamaParse 
parser = LlamaParse(
    api_key="YOUR_LLAMA_CLOUD_API_KEY",  
    result_type="markdown", 
    verbose=True,
    language="en",  
)


def get_section_nodes(docs):
    """Split each document into section nodes, by section headers marked with '#'. """
    nodes = []
    import re  
    min_length = 50
    for doc in docs:
        doc_sections = re.split(r'(?=\n# )', doc.text)  
        
        for section in doc_sections:
            if section.strip():  
                clean_section = section.strip()
                if len(clean_section) >= min_length:
                    node = TextNode(
                        text=clean_section,
                        metadata=deepcopy(doc.metadata),
                    )
                    nodes.append(node)
                
    return nodes


### Directory to load the papers from 
directory = './Final_papers/'

papers_dict = {}

### Loop over all pdf files (papers)
for filename in os.listdir(directory):

    file_path = os.path.join(directory, filename)
    base_filename = os.path.splitext(filename)[0]
 
    if file_path == "./Final_papers/.DS_Store":
        continue
    else: 
        documents = parser.load_data(file_path)
        
        page_nodes = get_section_nodes(documents)

        # Load the 
        path = f"./papers/pickled/{base_filename}.json"

        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        ### Extract the text from the second list
        text_in_second_list = {item['text'] for item in data}
            
        ### Filter the first list to only include nodes whose text is in the second list
        page_nodes = [node for node in page_nodes if node.text in text_in_second_list]
        
        ### Combine the text from all the nodes into a single text and store in a dictionary for all papers
        def combine_texts(text_nodes):
            
            text=''
            title = base_filename
            for node in text_nodes:
                text += node.text
            papers_dict[title] = text
            return papers_dict
        
        papers = combine_texts(page_nodes)
        
### Save all documents' and the full text in one document 
with open("papers_text.json", 'w', encoding='utf-8') as file:
    json.dump(papers, file, indent=4)
