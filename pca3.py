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

def pdf_read(pdf_doc):
    """Read the text from PDF document."""
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def create_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    """Create text chunks from a large text block."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    text_chunks = text_splitter.split_text(text)
    return text_chunks

def vector_store(text_chunks):
    """Create a vector store for the text chunks."""
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")

def query_llama_via_cli(input_text):
    """Query the Llama model via the CLI."""
    try:
        # Start the interactive process
        process = subprocess.Popen(
            ["ollama", "run", "llama3.1"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # Ensure that communication takes place as text (UTF-8)
            encoding='utf-8',  # Set UTF-8 encoding explicitly
            errors='ignore',  # Ignore incorrect characters
            bufsize=1
        )
        
        # Send the input to the process
        stdout, stderr = process.communicate(input=f"{input_text}\n", timeout=30)

        # Check error output
        if process.returncode != 0:
            return f"Error in the model request: {stderr.strip()}"

        # Filter response and remove control characters
        response = re.sub(r'\x1b\[.*?m', '', stdout)  # Remove ANSI codes

        # Extract the relevant answer
        return extract_relevant_answer(response)

    except subprocess.TimeoutExpired:
        process.kill()
        return "Timeout for the model request"
    except Exception as e:
        return f"An unexpected error has occurred: {str(e)}"

def extract_relevant_answer(full_response):
    """Extract the relevant response from the full model response."""
    response_lines = full_response.splitlines()

    # Search for the relevant answer; if there is a marker, it can be used here
    if response_lines:
        # Assume that the answer comes as a complete return to be filtered
        return "\n".join(response_lines).strip()

    return "No answer received"

def get_conversational_chain(context, ques):
    """Create the input for the model based on the prompt and context."""
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
    return response  

def user_input(user_question, pdf_text):
    context = pdf_text
    response = get_conversational_chain(context, user_question)  

    #st.markdown(f"""
    #<div style='background-color: #d1e7dd; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>
    #    <strong>User:</strong> {user_question}
    #</div>
    #""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style='background-color: #e2e3e5; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>
        <strong>PFD Assistant:</strong>
        <pre style="background-color: #282c34; color: white; padding: 10px;">{response}</pre>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main function of the Streamlit application."""
    st.set_page_config(page_title="CHAT WITH YOUR PDF")
    st.header("PDF CHAT APP")

    pdf_text = ""
    pdf_doc = st.file_uploader("Upload your PDF Files and confirm your question", accept_multiple_files=True)

    if pdf_doc:
        pdf_text = pdf_read(pdf_doc)  # Read the entire PDF text

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question and pdf_text:
        user_input(user_question, pdf_text)

    # Monitor RAM consumption
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 ** 2)  # Conversion to megabytes
    st.sidebar.write(f"Memory usage: {memory_usage:.2f} MB")

if __name__ == "__main__":
    main()
