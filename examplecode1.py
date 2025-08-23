import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama

from langchain.chains import SimpleSequentialChain

from langchain.memory import ConversationBufferMemory

from langchain.chains import SequentialChain


from pypdf import PdfReader
reader = PdfReader("example.pdf")
print(len(reader.pages))
