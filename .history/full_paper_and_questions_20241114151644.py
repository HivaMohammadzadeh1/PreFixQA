import json

### Load the paper text for all papers          
with open("papers_text.json", 'r', encoding='utf-8') as file:
    papers = json.load(file)
    
### Load the question answer pairs for all of papers 
with open("./qas.json", 'r', encoding='utf-8') as file:
    qas = json.load(file)
    
### List to hold the full dataset 
dataset = []

### Asssign and keep a context id to all papers from 0-(len(papers))
context_ids = {}
start = 0
for name in papers.keys():
    context_ids[name] = start
    start+=1 
    
### Asssign and keep the context lengths of all papers from 0-(len(papers))
context_lengths = {}
for name, text in papers.items():
    context_lengths[name] = len(text)
    
### Asssign and keep the question ids of all papers from 0-(len(papers)) and all of the questions 
question_ids = {}
for paper, qa_pairs in qas.items():
    question_ids[paper] = {}
    start = 0
    for question, answer in qa_pairs.items():
        question_ids[paper][question] = start
        start+=1 
    

Full_dataset = []
for name, text in papers.items():
    for paper, qa_pairs in qas.items():
        for question, answer in qa_pairs.items():
            if name == paper:
                dataset.append({"context": text, "context_id": context_ids[name], "question": question, "answer": answer, "question_id": question_ids[name][question], "context_length": context_lengths[name]}) 
        
### Save all documents' filtered (finalized) QA pairs and the text with assigned context and question ids to a single combined JSON file       
with open("prefix_qa_dataset.json", 'w', encoding='utf-8') as file:
    json.dump(dataset, file, indent=4)



papers_dataset = []
for name, text in papers.items():
    for paper, qa_pairs in qas.items():
        if name == paper:
            papers_dataset.append({"context": text, "context_id": context_ids[name], "context_length": context_lengths[name]}) 
            
### Save all documents and the text qith assigned context id and context lengths to a single combined JSON file       
with open("papers_dataset.json", 'w', encoding='utf-8') as file:
    json.dump(papers_dataset, file, indent=4)
