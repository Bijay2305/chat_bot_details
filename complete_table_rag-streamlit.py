import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai

# Initialize the embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index globally
index = None
combined_ticket_data = None

# Function to create an embedding for each ticket's important fields (flatten ticket info into a string)
def ticket_to_string(ticket_row):
    return f"Ticket {ticket_row['ticket_id']} assigned to {ticket_row['assigned_to']} solved by {ticket_row['solved_by']}"

# Function to retrieve relevant tickets from FAISS
def retrieve_relevant_ticket(query, top_k=5):
    query_embedding = embedder.encode([query], convert_to_tensor=False)
    
    # Search FAISS index
    distances, indices = index.search(np.array(query_embedding), top_k)
    
    # Retrieve the relevant tickets based on the indices
    relevant_tickets = combined_ticket_data.iloc[indices[0]]
    return relevant_tickets

# Function to generate a response using OpenAI GPT for more complex queries
def generate_gpt_response(query, relevant_tickets):
    # Create a prompt with relevant ticket information
    prompt = "Answer the following question based on the ticket details below:\n\n"
    for index, row in relevant_tickets.iterrows():
        prompt += f"Ticket {row['ticket_id']} assigned to {row['assigned_to']} was solved by {row['solved_by']}.\n"
    prompt += f"\nQuestion: {query}\nAnswer:"

    # Generate response from OpenAI
    response = openai.Completion.create(
        model="gpt-4",  # or "gpt-3.5-turbo"
        prompt=prompt,
        max_tokens=300,
        temperature=0.7,
        n=1
    )

    return response.choices[0].text.strip()

# Streamlit UI
st.title("Ticket Query System")

# Sidebar for OpenAI API Key
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

# Sidebar for Data Upload
uploaded_files = st.sidebar.file_uploader("Upload your CSV/Excel ticket data", type=['csv', 'xlsx'], accept_multiple_files=True)

if uploaded_files and openai_api_key:
    openai.api_key = openai_api_key
    all_data = []
    
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)
        else:
            st.warning("Unsupported file format. Please upload CSV or Excel files.")
            continue
        
        all_data.append(data)
    
    # Combine and clean data
    combined_ticket_data = pd.concat(all_data, axis=0).reset_index(drop=True)
    combined_ticket_data.columns = [col.strip().lower() for col in combined_ticket_data.columns]
    st.write("Data successfully loaded and combined.")
    
    # Show the data if needed
    if st.button("Show Ticket Data"):
        st.write(combined_ticket_data)
    
    # Convert tickets to embeddings and store in FAISS
    ticket_strings = combined_ticket_data.apply(ticket_to_string, axis=1).tolist()
    ticket_embeddings = embedder.encode(ticket_strings, convert_to_tensor=False)
    
    # Initialize FAISS index
    embedding_dim = len(ticket_embeddings[0])
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(ticket_embeddings))
    
    st.success("Data indexed for query retrieval!")
    
    # Query Input
    query = st.text_input("Enter your query (e.g., who solved ticket number XXXX?)")

    if query:
        relevant_ticket = retrieve_relevant_ticket(query)
        if not relevant_ticket.empty:
            st.write(f"Relevant Ticket(s):")
            st.write(relevant_ticket)
            gpt_response = generate_gpt_response(query, relevant_ticket)
            st.subheader("GPT Response:")
            st.write(gpt_response)
        else:
            st.write("No relevant ticket found.")
else:
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key.")
    elif not uploaded_files:
        st.warning("Please upload at least one CSV/Excel file containing ticket data.")
