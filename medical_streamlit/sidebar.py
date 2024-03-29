import streamlit as st 
import fitz
import tempfile
import os
import base64
from PyPDF2 import PdfReader 
import atexit
import requests 
import re
from bs4 import BeautifulSoup
from transformers import T5ForConditionalGeneration, AutoTokenizer, pipeline
import torch
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from langchain.chains.question_answering import load_qa_chain
import time


st.set_page_config(layout="wide")

# Load the model
@st.cache_resource
def load_medical_summarization_model():
    # Load the medical summarization model and tokenizer
    model_name = "Falconsai/medical_summarization"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Return the loaded model and tokenizer
    return model, tokenizer

model_name,tokenizer=load_medical_summarization_model()



def summary(text,max_length):

    summarizer = pipeline("summarization", model=model_name, tokenizer=model_name)

    # tokenize without truncation
    inputs_no_trunc = tokenizer(text, max_length=None, return_tensors='pt', truncation=False)

    # get batches of tokens corresponding to the exact model_max_length
    chunk_start = 0
    chunk_end = tokenizer.model_max_length  # == 1024 for Bart
    inputs_batch_lst = []
    while chunk_start <= len(inputs_no_trunc['input_ids'][0]):
        
        inputs_batch = inputs_no_trunc['input_ids'][0][chunk_start:chunk_end]  # get batch of n tokens
        inputs_batch = torch.unsqueeze(inputs_batch, 0)
        inputs_batch_lst.append(inputs_batch)
        chunk_start += tokenizer.model_max_length  # == 1024 for Bart
        chunk_end += tokenizer.model_max_length  # == 1024 for Bart


    # Manually construct custom prompt for the current batch
    custom_prompt = f"""Summarize the provided medical literature, extracting key findings, methodologies, and implications. 
    Tailor the summary to be accessible to healthcare professionals, researchers, medical students, and the general public. 
    Include essential details, important terms, and practical insights. 
    If the model identifies sections or paragraphs that it deems important based on the content, dynamically create new paragraphs with relevant headings to highlight these important insights. 
    Additionally, perform named entity recognition to identify and highlight important entities in the summary. Finally, provide a list of recognized key terms at the end of the summary : """

    # prompt=custom_prompt+inputs

    # generate a summary on each batch
    # summary_ids_lst = [model.generate(inputs, num_beams=4, max_length=max_length, early_stopping=True,) for inputs in inputs_batch_lst]
    
    # # decode the output and join into one string with one paragraph per summary batch
    # summary_batch_lst = []
    # for summary_id in summary_ids_lst:
    #     summary_batch = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_id]
    #     summary_batch_lst.append(summary_batch[0])
    # summary_all = '\n'.join(summary_batch_lst)



    summary_batch_lst = []
    for inputs in inputs_batch_lst:
        summary = summarizer(
            tokenizer.decode(inputs[0], skip_special_tokens=True),
            max_length=max_length//len(inputs_batch_lst),
            min_length=100,
            do_sample=False
        )[0]['summary_text']

        summary_batch_lst.append(summary)

    # join the summaries into one string
    summary_all = '\n'.join(summary_batch_lst)
    return summary_all



# Define a set of unwanted symbols
unwanted_symbols = {
    '✉', '❤', '⚠', '...', '<', '>', '!', '@', '#', '$', '%', '^', '&', '*',  '_', '+', '=', '[', ']', '{', '}',
    '|', '\\', ';', ':', '?', '/', '`', '~', '-', '\n', '\t', '\r'
} 

# Function to preprocess text
def preprocess_text(text):
    # Example: Remove non-alphanumeric characters and extra whitespaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Function to extract text from a website
def get_text_from_website(url):
    try:
        # Make a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad requests

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract text from the parsed HTML
        text = ' '.join([p.get_text() for p in soup.find_all('p')])

        # Preprocess the extracted text
        text = preprocess_text(text)
        print(len(text))

        return text
    except Exception as e:
        return f"Error: {e}"









def app():
    
    # Sidebar
    with st.sidebar:
        uploaded_file=st.file_uploader("Upload a PDF file", type=["pdf"])
        website_url=st.text_input("Enter the website URL:")
        max_length=st.slider('Maximum Length of the Summary', 100, 2000, 100)
        st.write(max_length)
        st.button("Button label")

    # Main content area
    col1, col2 = st.columns(2)

    # Inner containers closer to the sidebar
    with col1:
        # Container 1 for specific content
        container1=st.container(border=True,height=300)
        with container1:
            st.markdown("#### Article")
            # Place your content for container 1 here (text, images, etc.)
            st.write("""Content for container 1 near sid""")

        # Container 2 for additional content
        container2=st.container(border=True,height=300)
        with container2:
            st.markdown("#### Summary")
            # Place your content for container 2 here (text, charts, etc.)
            st.write("Content for container 2 near sidebar")
    with col2:
        
        st.write("Content for col2 2 near sidebar")
        st.header("Chat with PDF 💬")
        text=''
        if uploaded_file is not None:
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif website_url:
            text = get_text_from_website(website_url)

        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
        print('chunks')
        embeddings = HuggingFaceInstructEmbeddings(model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
        print('embeddign')
        vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
        print('vector')
        time.sleep(2)
        query = st.text_input("Question: ")
        if query:
            llm = HuggingFaceHub(repo_id="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract", model_kwargs={"temperature":0.5, "max_length":512})
            
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])


     

    # # Main content area logic
    # if uploaded_pdf:
    #     st.write("Uploaded PDF filename:", uploaded_pdf.name)
    #     # Add code to process or display the uploaded PDF content here

    # if website_url:
    #     st.write("You entered:", website_url)


    container2.write('sdfjashkdjfha')

    # if button_clicked:
    #     # Add code to handle the button click event here
    #     st.write("Button clicked!")

if __name__ == "__main__":
    app()
   