from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains.summarize import load_summarize_chain
from langchain_core.runnables import RunnablePassthrough


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
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


class RAG_Summary_QA:
    """Class to summarise the research paper and ask q&a"""
    
    def __init__(self,user_query, model, api_key, temperature):
        self.api_key = api_key
        self.model = model
        self.user_query = user_query
        self.temperature = temperature
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatGroq(api_key=self.api_key, model=self.model, temperature=self.temperature)
        self.parser = StrOutputParser()
        self.docs = ArxivLoader(query=self.user_query).load()
        self.final_docs = RecursiveCharacterTextSplitter(chunk_size=2500,chunk_overlap=100).split_documents(self.docs)

    def generate_summary(self):
        """Generates summary of research paper as the user enters research paper number."""
        
        # Add the support for summarization chain with map reduce or refine. Currently only being tested with page 1 content.
        text = self.final_docs[0].page_content
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
        chain = prompt | self.llm | self.parser
        response = chain.invoke({"input":user_query})
        return response

    def create_embeddings(self):
        vector_store = Chroma.from_documents(self.final_docs, embedding=self.embeddings, persist_directory='./embeddings')
        retriever = vector_store.as_retriever(search_kwargs={'k':2}, search_type='similarity')
        return retriever

    def get_answers_for_question(self, retriever, question):
        # ToDo : Add conversation history support.
        prompt = ChatPromptTemplate.from_template("""
        Answer the user input question based only on the provided contex. Make sure to keep the answer concise.
        Always add 'As per the given context' before answering the question.
        <question>
        {input}
        </question>
        <context>
        {context}
        </context>
        """
        )
        document_chain = create_stuff_documents_chain(llm=self.llm, prompt=prompt)
        retrival_chain = create_retrieval_chain(retriever,document_chain)
        response = retrival_chain.invoke(
            {
                "input":question
            }
        )
        print(response['context'])

        return response['answer']
    
if api_key:
    # [1706.03762] Attention Is All You Need
    if user_query:
        rag_qa = RAG_Summary_QA(user_query=user_query, api_key=api_key, model=selected_model_name, temperature=temperature)
        response = rag_qa.generate_summary()
        st.write(response)
        question_query = st.text_input("Ask your questions now.")
        if question_query:
            retriever = rag_qa.create_embeddings()
            qa_response = rag_qa.get_answers_for_question(retriever=retriever,question=question_query)
            st.write(qa_response)

    else:
        pass
    
else:
    st.warning('Please enter your groq api !')