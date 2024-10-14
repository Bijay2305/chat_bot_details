"""
Architecture Overview:
1. Data Sources
Instead of extracting from PDFs, you are working with structured data from various sources like databases, spreadsheets (CSV/Excel), or APIs.
The data likely includes columns such as ticket_id, assigned_to, status, solved_by, description, etc.
The system will need to handle multiple data sources, normalize them, and structure them into a centralized table.
2. Data Ingestion and Preprocessing
Load tabular data from multiple sources, process it to clean and normalize fields, and convert it into a unified DataFrame.
You can use pandas for this.
3. Embedding and Indexing
Use sentence-transformers or specialized table embedding models to create embeddings for each ticket or record.
Store these embeddings in FAISS for fast retrieval based on queries.
4. Query Processing and Retrieval
When a user enters a query (e.g., "Who solved ticket 1234?"), convert it into a query embedding and retrieve relevant records from FAISS.
5. Response Generation
For direct queries about ticket details, you can fetch the relevant information from the retrieved data.
Use OpenAI’s GPT only when more natural language processing is needed, such as combining or summarizing ticket details.

"""
"""
Step-by-Step Implementation
Ingest and Normalize Data from Different Sources
Here’s an example to load data from multiple sources like databases or CSV/Excel files."""
#step 1

import pandas as pd

# Example: Load data from different sources
ticket_data_1 = pd.read_csv('source_1_tickets.csv')
ticket_data_2 = pd.read_excel('source_2_tickets.xlsx')

# Combine them into a single DataFrame (unified format)
# Ensure consistent columns across sources (e.g., ticket_id, assigned_to, solved_by)
combined_ticket_data = pd.concat([ticket_data_1, ticket_data_2], axis=0).reset_index(drop=True)

# Clean and normalize data (e.g., handling missing values, fixing column names)
combined_ticket_data.columns = [col.strip().lower() for col in combined_ticket_data.columns]
combined_ticket_data.fillna("Unknown", inplace=True)

"""
step -2 Embedding and Indexing for Querying
Once the data is ingested and cleaned, you can create embeddings for each ticket's relevant information (like ticket_id, assigned_to, solved_by, etc.) and store them in a FAISS index."""
#step 2 
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Initialize the embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Function to create an embedding for each ticket's important fields (flatten ticket info into a string)
def ticket_to_string(ticket_row):
    return f"Ticket {ticket_row['ticket_id']} assigned to {ticket_row['assigned_to']} solved by {ticket_row['solved_by']}"

# Convert each ticket to a string representation
ticket_strings = combined_ticket_data.apply(ticket_to_string, axis=1).tolist()

# Create embeddings for each ticket
ticket_embeddings = embedder.encode(ticket_strings, convert_to_tensor=False)

# Store embeddings in FAISS index
embedding_dim = ticket_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(np.array(ticket_embeddings))
"""
step 3 Query Handling and Retrieval
Now, when a user asks a query like "Who solved ticket 1234?", we can retrieve relevant tickets using FAISS and return the most relevant ticket(s).
"""
# Function to retrieve relevant tickets from FAISS
def retrieve_relevant_ticket(query, top_k=5):
    query_embedding = embedder.encode([query], convert_to_tensor=False)
    
    # Search FAISS index
    distances, indices = index.search(np.array(query_embedding), top_k)
    
    # Retrieve the relevant tickets based on the indices
    relevant_tickets = combined_ticket_data.iloc[indices[0]]
    return relevant_tickets

# Example query
query = "Who solved the ticket number 1234?"
relevant_ticket = retrieve_relevant_ticket(query)

# Display relevant ticket information
print(relevant_ticket)
"""
step 4:Generating Responses Using OpenAI GPT
For more natural language processing queries or when a human-like response is needed, you can still use OpenAI’s GPT model. For simple fact-based queries like "Who solved ticket XXX?", you may just return the details directly from the data.
"""
import openai

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

# Example of GPT usage
gpt_response = generate_gpt_response(query, relevant_ticket)
print(gpt_response)

###++++++++++

"""
step 5Streamlit Application to Tie Everything Together
You can now create a Streamlit app to interact with the system, allowing users to upload data, query tickets, and get answers.
"""

import streamlit as st

st.title("Ticket Query System")

# Sidebar for Data Upload
uploaded_files = st.sidebar.file_uploader("Upload your CSV/Excel ticket data", type=['csv', 'xlsx'], accept_multiple_files=True)

if uploaded_files:
    all_data = []
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        all_data.append(data)
    
    # Combine and clean data
    combined_ticket_data = pd.concat(all_data, axis=0).reset_index(drop=True)
    combined_ticket_data.columns = [col.strip().lower() for col in combined_ticket_data.columns]

    # Show the data if needed
    if st.button("Show Ticket Data"):
        st.write(combined_ticket_data)

    # Query Input
    query = st.text_input("Enter your query (e.g., who solved ticket number XXXX?)")

    if query:
        relevant_ticket = retrieve_relevant_ticket(query)
        if not relevant_ticket.empty:
            st.write(f"Relevant Ticket:\n{relevant_ticket}")
            gpt_response = generate_gpt_response(query, relevant_ticket)
            st.write(f"GPT Response: {gpt_response}")
        else:
            st.write("No relevant ticket found.")
