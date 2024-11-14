import os 
import arxiv
from PyPDF2 import PdfReader


### Setup the Arxiv API and client
client = arxiv.Client()
search = arxiv.Search(
    query="Computer Science OR Biology OR Machine Learning OR Economics OR Computer Vision OR Electrical Engineering", #Topics
    max_results=300,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

### Loop through all the papers 
for result in client.results(search):
    paper = result
    # Fix the filename
    name = result.title.replace('/', '_').replace('\\', '_').replace(':', '_')
    
    # Specify the path to download the papers to 
    pdf_path = paper.download_pdf(dirpath="./papers", filename=f"{name}.pdf")

    # Initialize the PDF reader
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    pages_before_references = 0
    found_references = False

    # Count the number of pages before the references section
    for page_num in range(total_pages):
        page = reader.pages[page_num]
        text = page.extract_text()
        if text:
            text_lower = text.lower()
            if any(keyword in text_lower for keyword in ["references", "bibliography", "works cited", "literature cited"]):
                break
            else:
                pages_before_references += 1
        else:
            # If text extraction fails, assume the page is before References
            pages_before_references += 1


    # Make sure the paper includes at least 15 papers before its references
    if pages_before_references > 15:
        print(f"Paper '{name}' meets the page limit.")
    else:
        print(f"Paper '{name}' has {pages_before_references} pages which is less than the page limit and will be deleted.")
        os.remove(pdf_path)
