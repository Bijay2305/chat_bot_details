#Step 1: Document Processing and Embedding
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load the sentence transformer model for embeddings (pretrained model)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Function to load and split documents into chunks
def load_and_split_documents(doc_path, chunk_size=500):
    with open(doc_path, 'r') as f:
        text = f.read()
    # Split the text into chunks of chunk_size tokens
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Assume we have a directory of documents
documents_dir = 'documents/'
all_chunks = []

# Process all documents and split them into chunks
for doc_file in os.listdir(documents_dir):
    doc_path = os.path.join(documents_dir, doc_file)
    chunks = load_and_split_documents(doc_path)
    all_chunks.extend(chunks)

# Generate embeddings for all chunks
chunk_embeddings = embedder.encode(all_chunks, convert_to_tensor=False)

# Indexing using FAISS
embedding_dim = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(np.array(chunk_embeddings))

# Save the chunks and their embeddings for later use
chunk_data = {"chunks": all_chunks, "embeddings": chunk_embeddings}
