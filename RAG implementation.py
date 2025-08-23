import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

st.set_page_config(page_title="PDF Q&A (RAG)", page_icon="📄")
st.title("📄 PDF Q&A (RAG with Ollama)")

llm = Ollama(model="llama3.1:8b", temperature=0.2)
emb = OllamaEmbeddings(model="nomic-embed-text")

if "vs" not in st.session_state:
    st.session_state.vs = None

files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

def build_vectorstore(uploaded_files):
    docs = []
    for f in uploaded_files:
        reader = PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"source": f.name, "page": i + 1}
                ))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=120, separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, emb)

if files and st.button("Build knowledge base"):
    with st.spinner("Indexing..."):
        st.session_state.vs = build_vectorstore(files)
    st.success("Vector store ready!")

question = st.text_input("Ask a question about your PDFs")
if question and st.session_state.vs:
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=st.session_state.vs.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        verbose=True,
    )
    with st.spinner("Thinking..."):
        res = qa.invoke({"query": question})
    st.subheader("Answer")
    st.write(res["result"])

    st.subheader("Sources")
    for i, doc in enumerate(res["source_documents"], 1):
        st.write(f"{i}. {doc.metadata.get('source')} - page {doc.metadata.get('page')}")
else:
    st.info("Upload PDFs and click 'Build knowledge base' to start.")