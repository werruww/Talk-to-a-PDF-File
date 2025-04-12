import subprocess
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate
import os
import re
import psutil

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

embeddings = SpacyEmbeddings(model_name="en_core_web_sm")

def pdf_read(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")

def query_llama_via_cli(input_text):
    try:
        process = subprocess.Popen(
            ["ollama", "run", "llama3.1"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  
            encoding='utf-8',  
            errors='ignore',  
            bufsize=1
        )
        
        stdout, stderr = process.communicate(input=f"{input_text}\n", timeout=30)

        if process.returncode != 0:
            return f"Error in the model request: {stderr.strip()}"

        response = re.sub(r'\x1b\[.*?m', '', stdout) 
        return extract_relevant_answer(response)

    except subprocess.TimeoutExpired:
        process.kill()
        return "Timeout for the model request"
    except Exception as e:
        return f"An unexpected error has occurred: {str(e)}"

def extract_relevant_answer(full_response):
    response_lines = full_response.splitlines()

    if response_lines:
        return "\n".join(response_lines).strip()

    return "No answer received"

def get_conversational_chain(context, ques):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an intelligent and helpful assistant. Your goal is to provide the most accurate and detailed answers 
                possible to any question you receive. Use all available context to enhance your answers, and explain complex 
                concepts in a simple manner. If additional information might help, suggest further areas for exploration. If the 
                answer is not available in the provided context, state this clearly and offer related insights when possible.""",
            ),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    input_text = f"Prompt: {prompt.format(input=ques)}\nContext: {context}\nQuestion: {ques}"
    
    response = query_llama_via_cli(input_text)
    st.write("PDF: ", response)

def user_input(user_question, pdf_file):
    # Read the text from the selected PDF file
    pdf_text = pdf_read(pdf_file)
    
    # Create chunks from the text
    text_chunks = get_chunks(pdf_text)
    vector_store(text_chunks)
    
    combined_context = " ".join(text_chunks)
    get_conversational_chain(combined_context, user_question)

def main():
    st.set_page_config(page_title="CHAT WITH YOUT PDF")
    st.header("PDF CHAT APP")

    # Enable the upload of multiple PDF files
    pdf_docs = st.file_uploader(
        "Upload your PDF Files and confirm your question", 
        accept_multiple_files=True
    )

    if pdf_docs:
        # Create a list of file names
        pdf_names = [pdf.name for pdf in pdf_docs]
        # Enable the selection of a file
        selected_pdf_name = st.selectbox("Select a PDF to ask a question:", pdf_names)
        selected_pdf_file = next((pdf for pdf in pdf_docs if pdf.name == selected_pdf_name), None)

        user_question = st.text_input("Ask a Question from the Selected PDF File")

        if user_question and selected_pdf_file:
            user_input(user_question, selected_pdf_file)

    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 ** 2) 
    st.sidebar.write(f"Memory usage: {memory_usage:.2f} MB")

if __name__ == "__main__":
    main()
