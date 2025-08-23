import os
import requests
from langchain.llms.base import LLM
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import SequentialChain
import streamlit as st

# Custom LLM class using local Ollama (TinyLLaMA)
class OllamaLLM(LLM):
    model: str = "llama3.2"
    temperature: float = 0.8

    def _call(self, prompt: str, stop=None) -> str:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )
        if response.ok:
            return response.json()["response"]
        else:
            return f"Error: {response.status_code}"

    @property
    def _llm_type(self) -> str:
        return "ollama"

# Streamlit app
st.title('Celebrity Search Results')
input_text = st.text_input("Search the topic you want")

# Prompt Templates
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about celebrity {name}"
)

second_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="When was {person} born?"
)

third_input_prompt = PromptTemplate(
    input_variables=['dob'],
    template="Mention 5 major events that happened around {dob} in the world"
)

forth_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="What are {person} favourites crickets in the world he love to bat with"
)

# Memory
person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
descr_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')
best_memory = ConversationBufferMemory(input_key='dob', memory_key='best_memory')

# Ollama-based LLM
llm = OllamaLLM(temperature=0.8)
response = llm("Hello world how are you")  # Test the LLM to ensure it's working
print('******* Response from LLM:', response)
response = llm.predict('Who build you')  # Test the LLM to ensure it's working
print('******* Response from LLM predict:', response)

# Chains
chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True, output_key='person', memory=person_memory)
chain2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key='dob', memory=dob_memory)
chain3 = LLMChain(llm=llm, prompt=third_input_prompt, verbose=True, output_key='description', memory=descr_memory)
chain4 = LLMChain(llm=llm, prompt=forth_input_prompt, verbose=True, output_key='best_friends', memory=best_memory)

#SequentialChain - If we want to see all the chains we have created in sequence we use this
parent_chain = SequentialChain(
    chains=[chain, chain2, chain3, chain4],
    input_variables=['name'],
    output_variables=['person', 'dob', 'description', 'best_friends'],
    verbose=True
)

if input_text:
    result = parent_chain({'name': input_text})
    # result = parent_chain.run(input_text) run not supported when there is not exactly one output key. Got ['person','dob', 'description', 'best_friends']
    st.write(result)

    with st.expander('Person Name'):
        st.info(person_memory.buffer)

    with st.expander('Major Events'):
        st.info(descr_memory.buffer)

    with st.expander('Best Friends'):
        st.info(best_memory.buffer)