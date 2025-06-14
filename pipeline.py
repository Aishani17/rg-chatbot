import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import OPENAI_API_KEY
import openai

openai.api_key = OPENAI_API_KEY

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return " ".join([page.get_text() for page in doc])

def split_into_chunks(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def create_faiss_index(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, model, chunks

def retrieve_relevant_chunks(query, index, model, chunks, k=3):
    q_embed = model.encode([query])
    D, I = index.search(np.array(q_embed), k)
    return [chunks[i] for i in I[0]]

def ask_llm(context, question):
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Answer the question based on the given context."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']
