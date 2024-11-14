from openai import OpenAI
import re
from sklearn.cluster import KMeans
import json
import os
from openai import AzureOpenAI
from llama_parse import LlamaParse
from copy import deepcopy
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex

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


def combine_nodes(nodes):
    """Combine every two consecutive nodes."""
    combined_nodes = []
    i = 0
    while i < len(nodes):
        # Check if there are at least two nodes remaining to combine
        if i + 2 < len(nodes):
            combined_text = nodes[i].text + "\n" + nodes[i + 1].text
            combined_node = TextNode(
                text=combined_text,
                metadata=deepcopy(nodes[i].metadata),  
            )
            combined_nodes.append(combined_node)
            i += 2 
        else:
            combined_nodes.append(nodes[i])
            i += 1
    return combined_nodes
    
# Directory to load the papers from 
directory = './papers/Final_papers/'

### Store all of qa pairs and the filtered qa pairs for all papers 
all_papers_qa_pairs = {} 
all_papers_filtered_qa_pairs = {}

# Loop all pdf files 
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    base_filename = os.path.splitext(filename)[0]
    print(f'{base_filename}')
    print(file_path)
    if file_path == "./final_papers/.DS_Store":
        continue
    else: 
        documents = parser.load_data(file_path)
        
        page_nodes = get_section_nodes(documents)
        
        #Remove the bad nodes including the 'References' and 'ACKNOWLEDGEMENTS' sections
        path = f"./papers/pickled/{base_filename}.json"

        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Extract the text from the second list
        text_in_second_list = {item['text'] for item in data}
            
        # Filter the first list to only include nodes whose text is in the second list
        page_nodes = [node for node in page_nodes if node.text in text_in_second_list]

        combined_page_nodes = combine_nodes(page_nodes)

        #Filtered QA Pairs
        Final_qa_pairs = {}
        dicts_final = []
        
        #All QA Pairs
        All_qa_pairs = {}
        dicts_all = []

        #Hold the ratings for the questions and answers 
        All_qa_pairs_ratings = {}
        dicts_final_ratings = []

        ### Loop through all of the nodes 
        for i in range(len(combined_page_nodes)): 
            
            paragraph = combined_page_nodes[i].get_content()
            
            ###Generate the Question
            question = client.chat.completions.create(
                model="gpt-4-turbo-sehoon",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant tasked with generating very specific, \
                        unambiguous, short-answer questions based on a provided Document. \
                        The goal is to create high-quality synthetic data. Ensure the following: \
                        1. The question must be fact-based and directly answerable from the text of the provided Document and section of the paper. \
                        2. The question should not be vague, subjective, or open to interpretation. \
                        3. Focus on creating concise, precise questions that yield a 1-2 word answer. \
                        4. Questions should be relevant to the section of the paper and avoid overly broad phrasing. \
                        5. Avoid generating questions where the answer is too complex or requires long explanations. \
                        6. Make sure to clarify the model and dataset when asking a question specific to a model and dataset."},
                    {"role": "user", "content": f"Based on the section of the paper from the given document which is an arxiv paper, \
                        generate one short-answer question that asks for specific information retrievable directly from the section of the paper. \
                        The answer must be 1-2 words only. Follow this format from the example: \
                        Example: Question: What type of buffer is integrated with warp-specialization in FlashAttention-3? \
                        Use this section of the paper as the context: {paragraph}. \
                        DO NOT OUTPUT THE ANSWER; JUST THE QUESTION."}
                ],
                temperature=0.5
            )

            question1 = question.choices[0].message.content
            
            # Parse out the  leading "Question:" prefix
            match = re.search(r'^Question:', question1)
            if match: 
                question1 = re.sub(r'^Question:\s*', '', question1)

            answers_list = []

            ### Generate sample answers for the question 
            answers = client.chat.completions.create(
            model="gpt-4-turbo-sehoon",
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to generate accurate and specific \
                    short answers from a provided section of the paper and Document. \
                    Ensure the following: \
                    1. The answer must be concise (1-2 words). \
                    2. The answer must be directly retrieved from the provided text. \
                    3. If the section of the paper does not contain the information necessary to answer the question, respond with: 'The document does not contain the answer to the question.'. \
                    4. Avoid providing additional commentary, and only output the answer. \
                    5. If the answer is about a range, pay attention to the range and if there are negative signs."},
                {"role": "user", "content": f"Given the section of the paper below from an arXiv paper, generate a concise (1-2 words) answer to the following question. \
                    Retrieve the answer from the paper and the provided paragraph. PAY CLOSE ATTENTION TO THE Document AND RETRIEVE THE RIGHT ANSWER. \
                    OUTPUT THE ANSWER ONLY AND NOT THE QUESTION OR ANYTHING ELSE.\
                    Example: Question: What type of buffer is integrated with warp-specialization in FlashAttention-3? Answer: circular SMEM buffer. \
                    Here is the section of the paper: {paragraph}. \
                    Question: {question1}. Answer:"}
            ],
            temperature=1, 
            n=5
            )
            for choice in answers.choices:
                answers_list.append(choice.message.content)

            qa_all = {}
            qa_all[question1] = answers_list
           
            dicts_all.append(qa_all)

            ### Generate ratings for each answer 
            ratings = client.chat.completions.create(
                model = "gpt-4-turbo-sehoon",
                messages = [
                {"role": "system", "content": f"Please act as an impartial judge and evaluate the quality of the question and answer pairs provided. \
                    You will be assessing them based on the provided Document. Your evaluation should emphasize the following criteria:\
                    1. **Correctness**: Is the answer factually accurate and grounded in the document?\
                    2. **Agreement**: Does the answer directly address the question and provide a relevant response?\
                    3. **Confidence**: Does the answer confidently engage with the question, even if the Document does not contain the exact information? \
                    \
                    **Important considerations for rating:**\
                    - Rate vague or overly general questions lower, especially if they lack specificity or do not make sense in the context of the document.\
                    - Rate answers where the model, dataset, or method is unclear or missing details lower.\
                    - If the answer states that the information is not in the document, confirm by reviewing the document. If the information is indeed missing, rate the answer highly. If it is present, rate the answer lower.\
                    - Avoid focusing on questions about appendix numbers, or formatting details (like section names).\
                    - Avoid asking questions that have 2 possible answers. if there are 2 possible answers and only one is provided, rate the answers low.\
                    - If the answer is about a range, pay attention to the range and if there are negative signs. \
                    - Avoid asking questions that are taken from the 'References' and 'ACKNOWLEDGEMENTS' sections. if there are such questions, rate the answers low.\
                    \
                    For each answer, provide a rating on a scale from 1 to 10 based on its quality. Please output your rating in the following format:\
                    '[[rating]]', for example: 'Question: [question] Answer: [Answer]- Rating: [[5]].'\
                    ONLY OUTPUT THE QUESTION ONCE. Do NOT give explanations for your rating.\
                    Document: : {documents}"},
                {"role": "user", "content": f"The question is {question1} and the answers are {answers_list}. Make sure to give the rating for each answer in the answers list. \
                    Output in this format: Question: {question1} \
                    Answer: [Answer1]- Rating: [[5]]\
                    Answer: [Answer2]- Rating: [[8]] ..."}
            ]  
            )

            ratings = ratings.choices[0].message.content

            qa_all_ratings = {}
            qa_all_ratings[question1] = ratings
            dicts_final_ratings.append(qa_all_ratings)

            ### Only keep Questions where at least three out of five answers score above eight
            def parse_qa_with_high_ratings(data):
                
                
                lines = data.split('\n')
                
                qa_dict = {}
                current_question = None
                answers = {}  # Answers and their ratings
                high_rated_count = 0  # Counter for high-rated answers

                for line in lines:
                    if line.startswith("Question:"):
                        new_question = line[len("Question: "):].strip()
                        if new_question != current_question:
                            if current_question and high_rated_count >= 3:
                                maximum = max(answers, key=answers.get)
                                qa_dict[current_question] = answers[maximum]

                            current_question = new_question
                            answers = {}
                            high_rated_count = 0  

                    rating_match = re.search(r'\[\[([0-10]+)\]\]', line)
                    if rating_match:
                        rating = int(rating_match.group(1))
                        if rating >= 8:
                            answer = line.split('- Rating:')[0].strip()
                            answers[rating] = answer
                            high_rated_count += 1  #count for high-rated answers

                if current_question and high_rated_count >= 3:
                    maximum = max(answers, key=answers.get)
                    qa_dict[current_question] = answers[maximum]
                
                return qa_dict

            high_rated_qa = parse_qa_with_high_ratings(ratings)
            
            import re

            ### Clean the answer text
            def clean_answer(data):
                cleaned_data = {}
                for question, answer in data.items():
                    # Remove the "Answer: " prefix
                    answer = answer.replace("Answer: ", "")
                    # Remove trailing punctuation using regular expressions
                    
                    answer = re.sub(r'[^\w\s]', '', answer)
                    cleaned_data[question] = answer
                return cleaned_data

    
            high_rated_qa = clean_answer(high_rated_qa)
            
            dicts_final.append(high_rated_qa)

            print(high_rated_qa)
                
    
    for d in dicts_all:
        for k, v in d.items():  
            All_qa_pairs.setdefault(k, []).append(v)
        
        
    ### Save all of question answer pairs for each paper 
    encoding='utf-8'
    with open(f"qa_pairs_all_{base_filename}.json", 'w', encoding=encoding) as file:
        json.dump(All_qa_pairs, file, indent=4)
            
    for d in dicts_final:
        for k, v in d.items():  
            Final_qa_pairs.setdefault(k, []).append(v)

    ### Save Filtered question answer pairs for each paper 
    encoding='utf-8'
    with open(f"qa_pairs_filtered_{base_filename}.json", 'w', encoding=encoding) as file:
        json.dump(Final_qa_pairs, file, indent=4)
        
    all_papers_qa_pairs[base_filename] = All_qa_pairs
    all_papers_filtered_qa_pairs[base_filename]= Final_qa_pairs
    # print(all_papers_filtered_qa_pairs)
    

# Save all documents' all QA pairs to a single combined JSON file
with open("qa_pairs_combined.json", 'w', encoding='utf-8') as file:
    json.dump(all_papers_qa_pairs, file, indent=4)
    
# Save all documents' filtered (finalized) QA pairs to a single combined JSON file
with open("qa_pairs_combined_filtered.json", 'w', encoding='utf-8') as file:
    json.dump(all_papers_filtered_qa_pairs, file, indent=4)
