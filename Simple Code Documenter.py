import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama

# --- LLM and Chain Initialization ---
llm = Ollama(model="llama3.1:8b", temperature=0.2)

# --- Chain 1: Code Explanation ---
explanation_template = PromptTemplate(
    input_variables=["language", "code"],
    template="You are an expert programmer. Explain the following {language} code snippet in simple, easy-to-understand terms for a beginner.\n\nCode:\n```\n{code}\n```"
)
explanation_chain = LLMChain(llm=llm, prompt=explanation_template, output_key="explanation")

# --- Chain 2: Docstring Generation ---
docstring_template = PromptTemplate(
    input_variables=["language", "code"],
    template="You are an expert programmer. Generate a professional, well-formatted docstring for the following {language} code snippet. Only output the docstring itself, without any extra explanation.\n\nCode:\n```\n{code}\n```"
)
docstring_chain = LLMChain(llm=llm, prompt=docstring_template, output_key="docstring")

# --- Streamlit UI ---
st.title("🤖 Simple Code Documenter")
st.write("Paste your code below to get an explanation and a ready-to-use docstring.")

# List of common languages for the dropdown
LANGUAGES = ["Python", "JavaScript", "Java", "C++", "SQL", "Go", "Ruby", "TypeScript"]
selected_language = st.selectbox("Select the programming language:", LANGUAGES)

code_input = st.text_area("Paste your code here:", height=250, placeholder="def example_function(a, b):\n    return a + b")

if st.button("Generate Documentation") and code_input:
    with st.spinner("Thinking like a programmer..."):
        # Input dictionary for the chains
        input_data = {"language": selected_language, "code": code_input}

        # Run both chains
        explanation_result = explanation_chain.invoke(input_data)
        docstring_result = docstring_chain.invoke(input_data)

        st.subheader("Code Explanation")
        st.write(explanation_result["explanation"])

        st.subheader("Generated Docstring")
        # Use st.code to display the docstring with proper formatting
        st.code(docstring_result["docstring"], language=selected_language.lower())