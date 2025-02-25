import streamlit as st 
from PyPDF2 import PdfReader
import requests 
import re
from bs4 import BeautifulSoup
from transformers import T5ForConditionalGeneration, AutoTokenizer, pipeline
import torch
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from transformers import AutoModel, AutoTokenizer
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq



st.set_page_config(layout="wide")

# Load the model
@st.cache_resource
def load_medical_summarization_model():
    # Load the medical summarization model and tokenizer
    model_name = "Abdulkader/autotrain-medical-reports-summarizer-2484176581"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Return the loaded model and tokenizer
    return model_name, tokenizer

model_name,tokenizer=load_medical_summarization_model()


@st.cache_resource
def model_ans():
    # mdl='Maite89/Roberta_finetuning_semantic_similarity_stsb_multi_mt'
    mdl='sentence-transformers/all-mpnet-base-v2'
    # mdl='google/gemma-2-9b'
    # llm = GoogleGenerativeAI(model="gemini-pro", google_api_key='AIzaSyCCC9UN48NFIiTv4tTGb2SKR2HRHy93ZXk')
    llm=ChatGroq(model_name="llama3-8b-8192", groq_api_key="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")

    return mdl,llm

ans_model,llm=model_ans()

def summary(text, max_length):
  summarizer = pipeline("summarization", model=model_name, tokenizer=model_name)



  # Configure RecursiveCharacterTextSplitter (adjust chunk_size as needed)
  splitter = RecursiveCharacterTextSplitter(chunk_size=tokenizer.model_max_length - 2,
                                            chunk_overlap=100,)

  # Split the text into chunks
  chunks = splitter.split_text(text)

  summary_batch_lst = []
  for chunk in chunks:
    # Preprocess the chunk
    inputs_batch = tokenizer(chunk, return_tensors="pt", truncation=False)

    # Generate summary for the chunk with adjusted max_length
    summary = summarizer(
      tokenizer.decode(inputs_batch['input_ids'][0], skip_special_tokens=True),
      max_length=max_length // len(chunks),  # Adjust max_length per chunk
      min_length=100 // len(chunks),
      length_penalty=2.0,
      do_sample=False
    )[0]['summary_text']
    summary_batch_lst.append(summary)

  # Join the summaries into one string
  summary_all = '\n'.join(summary_batch_lst)
  return summary_all


# Define a set of unwanted symbols
unwanted_symbols = {
    '✉', '❤', '⚠', '...', '<', '>', '!', '@', '#', '$', '%', '^', '&', '*',  '_', '+', '=', '[', ']', '{', '}',
    '|', '\\', ';', ':', '?', '/', '`', '~', '-', '\n', '\t', '\r'
} 

# Function to preprocess text
def preprocess_text(text):
    # Remove small numbers (1-3 digits) that are not part of larger words or numbers:
    text = re.sub(r"(?<![a-zA-Z0-9])\d{1,3}(?![a-zA-Z0-9])", "", text)
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
    

def get_ans(texts,query):
    embeddings = HuggingFaceEmbeddings(model_name=ans_model)
        
    print('embeddign')
    db = FAISS.from_texts(texts, embeddings)
    print('vector')
    retriever=db.as_retriever(search_type='similarity_score_threshold', search_kwargs={"score_threshold": 0.5})
    docs = retriever.get_relevant_documents(query)
    return docs


def get_ans_web(texts,query):
    # texts=texts[0]
    print('type of the docs is : ',type(texts),'length of the docs: ',len(texts))
    embeddings = HuggingFaceEmbeddings(model_name=ans_model)
    print('embeddign')
    db = FAISS.from_texts(texts, embeddings)
    print('vector')
    embedding_vector = embeddings.embed_query(query)
    docs=db.similarity_search_by_vector(embedding_vector)
    print('type of the docs is : ',type(docs),'length of the docs: ',len(docs))
    return docs


def get_pdf_text(uploaded_file):
    text = ""
    pdf_reader = PdfReader(uploaded_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def cleaning(answer,question):
    
    template = """Thoroughly analyze the answer and check is this answer is correct answer of the question,
    respond the same answer with more detail only if the answer is correct answer of the question,
    if the answer is not the answer of the question don't try to make up an answer just respond as 'Unable to find the answer for that question from uploaded article',

    answer: {answer}

    Question: {question}

    Helpful Answer:"""
    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
    custom_rag_prompt
    | llm
    )

    output=rag_chain.invoke({'answer':answer,'question':question})
    return output





def app():
    
    # Sidebar
    with st.sidebar:
        uploaded_file=st.file_uploader("Upload a PDF file", type=["pdf"])
        website_url=st.text_input("Enter the website URL:")
        max_length=st.slider('Maximum Length of the Summary', 100, 2000, 100)
        st.write(max_length)
        summary_button=st.button("Summary")

    # Main content area
    col1, col2 = st.columns(2)

    # Inner containers closer to the sidebar
    with col1:
        container1=st.container(border=True,height=300)
        container1.markdown("#### Article")
        container2=st.container(border=True,height=300)
        container2.markdown("#### Summary")
        
        if uploaded_file is not None and summary_button:
        # Container 1 for specific content
            with container1:
                pdf_text = get_pdf_text(uploaded_file)
                preprocessed_text = preprocess_text(pdf_text)
                output = summary(preprocessed_text, max_length=max_length)
                st.write(preprocessed_text)
            with container2:
                st.write(output)

        elif website_url and summary_button:
            with container1:
                extracted_text = get_text_from_website(website_url)
                if "Error:" in extracted_text:
                    st.error(extracted_text)
                else:
                    st.write(extracted_text)
                    output=summary(extracted_text,max_length=max_length)
                    
            with container2:
                st.markdown("#### Summary")
                st.write(output)

    with col2:
        st.header("Ask About the ARTICLE")
        query = st.text_input("Question: ")
        container3=st.container(border=True,height=500)
        container3.header("Answer")

        text=''
        if uploaded_file is not None:
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            text=text.replace("\n", "")
            text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000,
            chunk_overlap=200,
            )
            texts = text_splitter.split_text(text)
            print(len(texts),type(texts))
            if query:
                print(query)
                docs=get_ans(texts,query)
                print('length of the docs<<<<<<<<<<<<<<<<<',len(docs))
                try:
                    ans=str(docs[0])
                    ans=ans.replace('page_content=', '')
                    anss=cleaning(ans,query)
                    if hasattr(anss, "content"):
                        container3.write(anss.content)
                    else:
                        container3.write("No valid response generated.")
                except IndexError:
                    container3.write('Unable to find the answer for that question from uploaded article')

                # ans=str(docs[0])
                # ans=ans.replace('page_content=', '')
                # anss=cleaning(ans,query)
                # container3.write(anss)




        elif website_url:
            text=""
            text = get_text_from_website(website_url)
            print(len(text))
            text_splitter = RecursiveCharacterTextSplitter(
            # separators=['\n\n', '\n', '.', ','],
            chunk_size=1000,
            chunk_overlap=200,
            )
            texts=text_splitter.split_text(text)
            print(len(texts),type(texts))
            if query:
                print("link query",query)
                wdocs=get_ans_web(texts,query)
                print('<<<<<<<<<<<<<<<<<',len(wdocs))
                try:
                    strr=str(wdocs[0])
                    strr=strr.replace('page_content=', '')
                    strrr=cleaning(strr,query)
                    if hasattr(strrr, "content"):
                        container3.write(strrr.content)
                    else:
                        container3.write("No valid response generated.")
                except IndexError:
                    container3.write('Unable to find the answer for that question from uploaded article')


                

        
        



if __name__ == "__main__":
    app()
   
