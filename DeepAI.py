import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai
from dotenv import load_dotenv
load_dotenv()
import os

# Initialize Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize OpenAI API (replace with your API key)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to chunk text into sections
def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Function to create embeddings and build FAISS index
def create_faiss_index(chunks):
    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, embeddings

# Function to perform semantic search
def semantic_search(query, index, embeddings, chunks, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

# Function to generate response using OpenAI GPT
def generate_response(query, relevant_chunks):
    context = " ".join(relevant_chunks)
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7
    )
    return response.choices[0].text.strip()

# Streamlit UI
st.title("DeepDive AI - Research Paper Insights")

# File upload
uploaded_file = st.file_uploader("Upload a research paper (PDF)", type="pdf")

if uploaded_file is not None:
    # Extract text from PDF
    text = extract_text_from_pdf(uploaded_file)
    
    # Chunk text
    chunks = chunk_text(text)
    
    # Create FAISS index
    index, embeddings = create_faiss_index(chunks)
    
    # User query
    query = st.text_input("Enter your question about the paper:")
    
    if query:
        # Perform semantic search
        relevant_chunks = semantic_search(query, index, embeddings, chunks)
        
        # Generate response
        response = generate_response(query, relevant_chunks)
        
        # Display response
        st.subheader("Answer:")
        st.write(response)
        
        # Display relevant sections
        st.subheader("Relevant Sections from the Paper:")
        for i, chunk in enumerate(relevant_chunks):
            st.write(f"Section {i+1}:")
            st.write(chunk)