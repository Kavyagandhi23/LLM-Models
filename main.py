# Integrate code with OpenAi API
import os
from constants import openai_key
from langchain.llms import OpenAI
import streamlit as st
os.environ["OPENAI_API_KEY"] = openai_key

# Initialize streamlit framework
st.title('LLM model Using OpenAI')
input = st.text_input("Search for any topic")

# OpenAI LLM model
llm=OpenAI(temperature=0.8)

if input:
    st.write(llm(input))