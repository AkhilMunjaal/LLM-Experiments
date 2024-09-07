from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import ArxivLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

## set up Streamlit 
st.title("Conversational RAG With Research Papers !")

## LLM Control Parameters for User
api_key=st.sidebar.text_input("Enter your Groq API key:",type="password")
selected_model_name=st.sidebar.selectbox("Select the open source model : ",["gemma2-9b-it","llama-3.1-8b-instant","mixtral-8x7b-32768"])
temperature = st.sidebar.slider("Temperature : ", min_value=0.0, max_value=1.0, value=0.5)
max_tokens = st.sidebar.slider("Max output tokens : ", min_value=50, max_value=3000, value=1500)
user_query = st.text_input("Enter the research paper number")

def generate_summary(user_query, model, api_key, temperature, max_tokens):
    """Generates summary of research paper as the user enters research paper number."""
    llm = ChatGroq(api_key=api_key, model=model, temperature=temperature)
    parser = StrOutputParser()
    docs = ArxivLoader(query=user_query).load()
    text = docs[0].page_content[:2000]
    print(llm.get_num_tokens(docs[0].page_content))
    prompt = ChatPromptTemplate.from_template(
        f"""
        You are an AI expert researcher with immense understanding of research papers.
        When presented a huge corpus of text, summarize it in a concise manner with the following format:
        Title: [Title of the Paper]
        Authors: [List of Authors]
        Summary: [Brief Summary]
        
        {text}
        """
    )
    chain = prompt | llm | parser
    response = chain.invoke({"input":user_query})
    return response
    
if api_key:
    if user_query:
        response = generate_summary(user_query=user_query, api_key=api_key, model=selected_model_name, temperature=temperature, max_tokens=max_tokens)
        st.write("Summary :")
        st.write(response)
        st.text_input("Ask your questions now.")
    else:
        pass
    
else:
    st.warning('Please enter your groq api !')