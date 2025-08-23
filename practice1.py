import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_community.llms import Ollama

# Initialize the LLM
llm = Ollama(model="llama3.1:8b", temperature=0.2)

# Streamlit UI
st.title("🎬 Celebrity Search App")
input_text = st.text_input("Enter the name of a celebrity:")

# First prompt template
prompt_template = PromptTemplate(
    input_variables=["name"],
    partial_variables={"tone": "informative"},
    template="Give a short biography and recent news about the celebrity {name} in this {tone} way."
)
chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True, output_key="person")

# Second prompt template
second_prompt_template = PromptTemplate(
    input_variables=["person"],
    template="When was {person} born?"
)
chain2 = LLMChain(llm=llm, prompt=second_prompt_template, verbose=True, output_key="DOB")

# Sequential chain
parent_chain = SequentialChain(
    chains=[chain, chain2],
    input_variables=["name"],     # first chain's input
    output_variables=["person", "DOB"],  # final outputs
    verbose=True
)

# Handle input and generate output
if input_text:
    result = parent_chain({"name": input_text})
    st.write("Biography and News:")
    st.write(result["person"])
    st.write("Date of Birth:")
    st.write(result["DOB"])
