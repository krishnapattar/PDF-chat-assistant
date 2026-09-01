import streamlit as st
from rag.loader import load_pdf_text
from rag.chunker import chunk_text
from rag.vector_store import add_chunks, reset_collection
from rag.qa_chain import answer_question

st.title("Chat With Your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    if "processed" not in st.session_state:
        with open("data/sample.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Reading and indexing your PDF..."):
            reset_collection()
            text = load_pdf_text("data/sample.pdf")
            chunks = chunk_text(text)
            add_chunks(chunks)    


        st.session_state["processed"] = True
        st.success(f"Indexed {len(chunks)} chunks. Ask away!")

    question = st.text_input("Ask a question about your PDF")

    if question:
        with st.spinner("Thinking..."):
            answer = answer_question(question)
        st.write(answer)