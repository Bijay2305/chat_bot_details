#pip install streamlit openai sentence-transformers faiss-cpu
#step 2 - Set up your OpenAI API key by creating a .env file and adding your key:
OPENAI_API_KEY=your-api-key-here

#You can load this in your Python code using the dotenv library:
pip install python-dotenv

#app.py
import os
import streamlit as st
import openai
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load OpenAI API key from environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set up Streamlit
st.title("RAG System: Document-Based Question Answering")

# Initialize the Sentence Transformer model for embedding and FAISS index
embedder = SentenceTransformer('all-MiniLM-L6-v2')
index = None  # FAISS index will be initialized later
all_chunks = []  # Store document chunks

# Sidebar to upload documents
st.sidebar.title("Document Upload")
uploaded_files = st.sidebar.file_uploader("Upload your text files", type="txt", accept_multiple_files=True)

# Function to split documents into chunks
def split_into_chunks(document_text, chunk_size=500):
    return [document_text[i:i + chunk_size] for i in range(0, len(document_text), chunk_size)]

# Step 1: Load documents, split into chunks, and create embeddings
if uploaded_files:
    all_chunks = []
    
    for uploaded_file in uploaded_files:
        # Read text file
        document_text = uploaded_file.read().decode('utf-8')
        chunks = split_into_chunks(document_text)
        all_chunks.extend(chunks)
    
    st.sidebar.success(f"Loaded {len(uploaded_files)} documents and split into {len(all_chunks)} chunks.")
    
    # Step 2: Create embeddings and store in FAISS index
    chunk_embeddings = embedder.encode(all_chunks, convert_to_tensor=False)
    
    # Initialize FAISS index
    embedding_dim = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(chunk_embeddings))

# Step 3: User query input
st.header("Ask a Question Based on the Uploaded Documents")
user_query = st.text_input("Enter your query:")

# Function to retrieve similar document chunks using FAISS
def retrieve_similar_chunks(query, top_k=5):
    query_embedding = embedder.encode([query], convert_to_tensor=False)
    
    # Search the FAISS index for the top_k similar chunks
    distances, indices = index.search(np.array(query_embedding), top_k)
    
    # Retrieve the actual chunks based on their indices
    similar_chunks = [all_chunks[i] for i in indices[0]]
    return similar_chunks

# Function to generate a response using OpenAI GPT
def generate_gpt_response(user_query, retrieved_chunks):
    # Create prompt with context from retrieved chunks
    prompt = "Answer the following question based on the context below:\n\n"
    for chunk in retrieved_chunks:
        prompt += f"{chunk}\n\n"
    prompt += f"Question: {user_query}\nAnswer:"
    
    # Use the OpenAI GPT API to generate a response
    response = openai.Completion.create(
        model="gpt-4",  # You can also use "gpt-3.5-turbo"
        prompt=prompt,
        max_tokens=300,
        temperature=0.7,
        n=1
    )
    
    return response.choices[0].text.strip()

# Step 4: Display query results
if user_query and index is not None:
    with st.spinner("Retrieving relevant document chunks..."):
        # Retrieve similar chunks from FAISS
        similar_chunks = retrieve_similar_chunks(user_query)
        
        # Generate response using OpenAI GPT
        gpt_response = generate_gpt_response(user_query, similar_chunks)
        
        # Display the response
        st.subheader("Generated Response:")
        st.write(gpt_response)
        
        # Display the retrieved document chunks
        st.subheader("Relevant Document Chunks:")
        for i, chunk in enumerate(similar_chunks, start=1):
            st.markdown(f"**Chunk {i}:** {chunk}")
else:
    if not uploaded_files:
        st.write("Please upload text files to proceed.")
    elif not user_query:
        st.write("Enter a query to retrieve relevant information from the uploaded documents.")

#run the application 

Run the Streamlit app
http://localhost:8501

 