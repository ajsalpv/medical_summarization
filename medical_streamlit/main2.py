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
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from langchain.chains.question_answering import load_qa_chain
import time


# Load the model
model_name = "Falconsai/medical_summarization"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


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


# ChatBot
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            
            with st.chat_message("user"):
                st.write(message.content)
        else:
            with st.chat_message("assistant"):
                st.write(message.content)
    return {"assistant_response": response['assistant_response']}
    

def main():

    load_dotenv()

    # Set page configuration to wide layout
    st.set_page_config(layout="wide")

    with st.chat_message("user"):
        st.write('prompt')


    max_length = st.slider('Maximum Length of the Summary', 100, 2000, 100)
    st.write(max_length)

    # File uploader for PDF files
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    # Input box for entering the website URL
    website_url = st.text_input("Enter the website URL:")


        # Column 1 content (placeholder for now)

    if uploaded_file is not None or website_url:
            
            if st.button("Summarization"):
                st.header("Summarization")
                # Create two columns
                col1, col2 = st.columns([1,1])

                if uploaded_file is not None:
                    with col1:
                        st.write("Column 1 content will go here")
                        # Create a temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                            temp_file_path = temp_file.name
                            temp_file.write(uploaded_file.getvalue())

                            # Open the saved PDF file using PyMuPDF
                            pdf_document = fitz.open(temp_file_path)

                            # Define the translation table to remove unwanted symbols
                            translation_table = str.maketrans('', '', ''.join(unwanted_symbols))

                            # Iterate through pages
                            for page_num in range(pdf_document.page_count):
                                page = pdf_document[page_num]
                                text = page.get_text()

                                # Check if text contains "REFERENCE" in uppercase
                                if "REFERENCE" in text.upper():
                                    # Stop appending text when reference is found
                                    break

                                # Remove unwanted symbols
                                text = text.translate(translation_table)

                                # Display text content in Streamlit
                                st.write(text)
                                output=summary(text,max_length=700)

                            # Close the PDF document
                            pdf_document.close()

                            time.sleep(5)
                            # Remove the temporary file
                            os.remove(temp_file_path)

                    with col2:
                        st.write("Summary")
                        st.write(output)

                        
                elif website_url:
                    with col1:
                        st.write("Column 1 content will go here")
                        extracted_text = get_text_from_website(website_url)
                        st.write("Extracted and Preprocessed Text:")
                        if "Error:" in extracted_text:
                            st.error(extracted_text)
                        else:
                            st.write(extracted_text)
                            output=summary(extracted_text,max_length=max_length)

                    with col2:
                        st.write("Summary")
                        st.write(output)


            if st.button("ChatBot"):
                
                st.header("ChatBot")



                with st.spinner("Processing"):


                    st.header("Chat with PDF 💬")
                    if "messages" not in st.session_state.keys():
                        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

                        # Display or clear chat messages
                        for message in st.session_state.messages:
                            with st.chat_message(message["role"]):
                                st.write(message["content"])
    
    
                    
                
                    # st.write(pdf)
                    if uploaded_file is not None:
                        pdf_reader = PdfReader(uploaded_file)
                        
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text()
                    elif website_url:
                        text = get_text_from_website(website_url)
                        
                
                    text_splitter = CharacterTextSplitter(
                        separator="\n",
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len
                        )
                    chunks = text_splitter.split_text(text=text)

                    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

                    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
                
                        
                    # query = st.text_input("Ask questions about your PDF file:")
                    # # st.write(query)
                
                    # if query:
                    #     docs = vectorstore.similarity_search(query=query, k=3)
                
                    #     llm = "hkunlp/instructor-large"
                    #     chain = load_qa_chain(llm=llm, chain_type="stuff")
                            
                    #     response = chain.run(input_documents=docs, question=query)
                    #     st.write(response)

                    def generate_response(query):
                        docs = vectorstore.similarity_search(query=query, k=3)
                
                        llm = "hkunlp/instructor-large"
                        chain = load_qa_chain(llm=llm, chain_type="stuff")
                            
                        response = chain.run(input_documents=docs, question=query)
                        print(response)
                        return response
                        



                    if prompt := st.chat_input("Ask a question about your documents:"):
                        st.session_state.messages.append({"role": "user", "content": prompt})
                        with st.chat_message("user"):
                            st.write(prompt)


                    # Generate a new response if last message is not from assistant
                    if st.session_state.messages[-1]["role"] != "assistant":
                        with st.chat_message("assistant"):
                            with st.spinner("Thinking..."):
                                response = generate_response(prompt)
                                placeholder = st.empty()
                                full_response = ''
                                for item in response:
                                    full_response += item
                                    placeholder.markdown(full_response)
                                placeholder.markdown(full_response)
                        message = {"role": "assistant", "content": full_response}
                        st.session_state.messages.append(message)
                                    
                               
                    



if __name__ == '__main__':
    main()

                




        


       
           
            















   
    




