#Step 2: Query and Retrieval
# Function to process a user query and retrieve the most similar document chunks
def retrieve_similar_chunks(query, top_k=5):
    query_embedding = embedder.encode([query], convert_to_tensor=False)
    
    # Search in the FAISS index for similar chunks
    distances, indices = index.search(np.array(query_embedding), top_k)
    
    # Retrieve the top_k most similar chunks
    similar_chunks = [all_chunks[i] for i in indices[0]]
    return similar_chunks

