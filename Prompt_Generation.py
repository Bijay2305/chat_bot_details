#Step 3: Prompt Generation and LLM Response
#Here, we use a large language model (LLM) like GPT-3 or GPT-4 from the transformers library to generate responses.
from transformers import pipeline

# Load a pre-trained LLM (like GPT-4 or any available model)
generator = pipeline('text-generation', model='gpt-3.5-turbo')

# Function to generate the prompt and get the response from the LLM
def generate_response(user_query, retrieved_chunks):
    # Format the prompt
    prompt = f"Answer the question based on the following context:\n\n"
    for chunk in retrieved_chunks:
        prompt += f"{chunk}\n\n"
    prompt += f"Question: {user_query}\nAnswer:"
    
    # Use the LLM to generate a response
    response = generator(prompt, max_length=200, num_return_sequences=1)
    return response[0]['generated_text']

# Example usage
user_query = "What are the main benefits of using a RAG system?"
similar_chunks = retrieve_similar_chunks(user_query)
response = generate_response(user_query, similar_chunks)
print(response)
