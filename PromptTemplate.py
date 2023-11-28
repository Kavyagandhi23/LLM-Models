import os
from langchain import OpenAI
from constants import openai_key
import streamlit as st
os.environ['OPENAI_API_KEY']=openai_key

from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.memory import ConversationBufferMemory

st.title('LLM Model for Restaurant Search')
input_text = st.text_input('Search for any restaurant for specific country')

# Initialize LLM
llm = OpenAI(temperature=0.8)

# Memory
Restaurant_memory = ConversationBufferMemory(input_key='Country', memory_key = 'chat_history')
Cuisine_memory = ConversationBufferMemory(input_key='Restaurants Name', memory_key = 'chat_history')

# Prompt Template
first_prompt = PromptTemplate(
    input_variables=['Country'],
    template='Give the list of top 5 Restaurants of {Country}'
)
chain = LLMChain(llm=llm, prompt = first_prompt, verbose = True, output_key='Restaurants Name', memory=Restaurant_memory)

second_prompt = PromptTemplate(
    input_variables=['Restaurants Name'],
    template='Give the cuisine of {Restaurants Name}'
)
chain2 = LLMChain(llm=llm, prompt = second_prompt, verbose = True, output_key='cuisine',memory=Cuisine_memory)

parent_chain = SimpleSequentialChain(
    chains=[chain,chain2], verbose = True)


if input_text:
    st.write(parent_chain.run(input_text))

    with st.expander('Country Name'):
        st.info(Restaurant_memory.buffer)

    with st.expander('Cuisine Memory'):
        st.info(Cuisine_memory.buffer)