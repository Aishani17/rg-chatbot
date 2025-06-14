import streamlit as st
from pipeline import *

st.title("📄 Chat with your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
question = st.text_input("Ask a question")

if uploaded_file and question:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Processing..."):
        text = extract_text_from_pdf("temp.pdf")
        chunks = split_into_chunks(text)
        index, model, chunk_data = create_faiss_index(chunks)
        relevant_chunks = retrieve_relevant_chunks(question, index, model, chunk_data)
        context = "\n\n".join(relevant_chunks)
        answer = ask_llm(context, question)

    st.success("💬 Answer:")
    st.write(answer)
