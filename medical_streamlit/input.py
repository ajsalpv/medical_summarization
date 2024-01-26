import streamlit as st
from langchain_community.document_loaders import PyPDFLoader



# Upload file
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    with open(uploaded_file.name, "rb") as f:
        loader = PyPDFLoader(f)
        documents = loader.load()
if documents:
    st.write('docu')