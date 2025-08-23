import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_community.llms import Ollama

# --- LLM and Chain Initialization ---
# Use a higher temperature for more creative outputs
llm = Ollama(model="llama3.1:8b", temperature=0.7)

# --- Streamlit UI ---
st.title("✍️ Creative Writer's Toolkit")

topic = st.text_input("What is the story about? (e.g., 'a lost robot in a fantasy forest')")
genre = st.selectbox("Select a genre:", ["Fantasy", "Science Fiction", "Mystery", "Adventure", "Horror"])

# --- Chain 1: Story Generation ---
story_template = PromptTemplate(
    input_variables=["topic", "genre"],
    template="Write a short, engaging story (about 200 words) in the {genre} genre about {topic}."
)
story_chain = LLMChain(llm=llm, prompt=story_template, output_key="story_text", verbose=True)

# --- Chain 2: Title Generation ---
# This chain will take the output from the first chain ("story_text") as its input
title_template = PromptTemplate(
    input_variables=["story_text"],
    template="Based on the following story, suggest 3 creative and fitting titles:\n\n{story_text}"
)
title_chain = LLMChain(llm=llm, prompt=title_template, output_key="titles", verbose=True)

# --- Sequential Chain to connect them ---
# The output of story_chain ('story_text') is automatically fed as input to title_chain
parent_chain = SequentialChain(
    chains=[story_chain, title_chain],
    input_variables=["topic", "genre"],  # Inputs for the very first chain
    output_variables=["story_text", "titles"], # Final desired outputs
    verbose=True
)

if topic and genre:
    if st.button("Generate Story & Titles"):
        with st.spinner("Crafting a masterpiece..."):
            result = parent_chain.invoke({"topic": topic, "genre": genre})

            st.subheader("Your Generated Story")
            st.write(result["story_text"])

            st.subheader("Suggested Titles")
            st.write(result["titles"])