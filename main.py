from fastapi import FastAPI
import chromadb
from chromadb.config import Settings
import numpy as np
import torch
import cohere

app = FastAPI()
co = cohere.Client(f"<<COHERE_API_KEY>>")

# Initialize the Chroma client
client = chromadb.Client(Settings(
    chroma_db_impl="deta",
    persist_directory="./chroma-data"
))

# Get the Wikipedia collection
collection = client.get_collection("wikipedia")

@app.get("/search")
def search_wikipedia(query: str, top_k: int = 10):
    # Encode the query using the Cohere model
    response = co.embed(texts=[query], model='multilingual-22-12')
    query_embedding = torch.tensor(response.embeddings[0])

    # Search the Chroma collection for the top k results
    results = collection.query(
        query_embeddings=query_embedding.numpy(),
        n_results=top_k
    )

    # Format the results
    output = []
    for i in range(top_k):
        doc_id = results.ids[0][i]
        doc_title = results.metadatas[0][doc_id]['title']
        doc_text = results.documents[0][i]
        output.append({
            'id': doc_id,
            'title': doc_title,
            'text': doc_text
        })

    return output

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)