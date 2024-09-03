from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
import streamlit as st
import os

def generate_response(question, model, temperature, max_tokens):
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI assistant who can answer the user query."),
        ("user", question)
    ]
)
    parser = StrOutputParser()
    llm = ChatOllama(model=model, temperature=temperature,num_predict=max_tokens)
    chain = prompt | llm | parser
    response = chain.invoke({"question":question})
    return response

st.title("Basics QA with Streamlit.")

engine = st.sidebar.selectbox("Select the open source model : ",["mistral","llama3"])
temperature = st.sidebar.slider("Temperature : ", min_value=0.0, max_value=1.0, value=0.5)
max_tokens = st.sidebar.slider("Max output tokens : ", min_value=50, max_value=300, value=150)

st.write("Ask you question to any open source LLM")
user_input = st.text_input("User: ")


if user_input:
    response = generate_response(model=engine,question=user_input,temperature=temperature,max_tokens=max_tokens)
    st.write(response)
else:
    st.write("Please ask your question!")