import os
import tempfile
from typing import TypedDict, List

import streamlit as st
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


# ---------------------------------
# Load environment variables
# ---------------------------------
load_dotenv()


# ---------------------------------
# Gemini models
# ---------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)


# ---------------------------------
# Streamlit page
# ---------------------------------
st.set_page_config(page_title="Multi PDF RAG")
st.title("Multi-PDF RAG Chatbot")


# ---------------------------------
# Upload PDFs
# ---------------------------------
uploaded_files = st.file_uploader(
    "Upload one or more PDFs",
    type="pdf",
    accept_multiple_files=True
)


# ---------------------------------
# Build vector DB from uploaded PDFs
# ---------------------------------
def build_vectorstore(uploaded_files):
    documents = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        for d in docs:
            d.metadata["source"] = uploaded_file.name

        documents.extend(docs)

        os.remove(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    vectorstore.persist()

    return vectorstore


# ---------------------------------
# LangGraph state
# ---------------------------------
class GraphState(TypedDict):
    question: str
    documents: List
    answer: str


# ---------------------------------
# Build LangGraph app
# ---------------------------------
def build_graph(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def retrieve(state):
        docs = retriever.invoke(state["question"])
        return {"documents": docs}

    def generate(state):
        question = state["question"]
        docs = state["documents"]

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
Answer ONLY using the provided context.
If the answer is not in the context, say you do not know.

Context:
{context}

Question:
{question}
"""

        response = llm.invoke(prompt)

        return {"answer": response.content}

    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)

    workflow.set_entry_point("retrieve")

    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


# ---------------------------------
# Build DB button
# ---------------------------------
if uploaded_files:
    if st.button("Process PDFs"):
        with st.spinner("Reading PDFs and building vector database..."):
            vectorstore = build_vectorstore(uploaded_files)
            st.session_state["vectorstore"] = vectorstore
            st.session_state["graph"] = build_graph(vectorstore)

        st.success("PDFs processed successfully.")


# ---------------------------------
# Ask questions
# ---------------------------------
if "graph" in st.session_state:
    question = st.text_input("Ask a question about the PDFs")

    if question:
        graph = st.session_state["graph"]

        result = graph.invoke({
            "question": question
        })

        st.subheader("Answer")
        st.write(result["answer"])
      
