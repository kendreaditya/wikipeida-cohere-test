from datasets import load_dataset
import chromadb
from chromadb.config import Settings
import os

# Load the Wikipedia dataset with embeddings
docs = load_dataset("Cohere/wikipedia-22-12-en-embeddings", split="train", streaming=True)

# Initialize the Chroma client
client = chromadb.Client(Settings(
    chroma_db_impl="deta",
    persist_directory="./chroma-data"
))

# Create a new collection
collection = client.create_collection("wikipedia")

# Iterate through the documents and add them to the Chroma collection
for doc in docs:
    docid = doc['id']
    title = doc['title']
    text = doc['text']
    emb = doc['emb']
    collection.add(
        ids=[docid],
        documents=[text],
        metadatas=[{'title': title}],
        embeddings=[emb]
    )

print("Wikipedia dataset loaded into Chroma DB.")