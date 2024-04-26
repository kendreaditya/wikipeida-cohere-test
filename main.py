from fastapi import FastAPI
import numpy as np
import cohere
import diskannpy
from datasets import load_dataset

app = FastAPI()
co = cohere.Client(f"<<COHERE_API_KEY>>")

docs = load_dataset("Cohere/wikipedia-22-12-en-embeddings", split="train")

index_dir = "./wikipedia-index"

# To search the index:
index = diskannpy.StaticDiskIndex(index_dir, num_threads=4)

@app.get("/search")
def search_wikipedia(query: str, top_k: int = 10):
    # Encode the query using the Cohere model
    response = co.embed(texts=[query], model='multilingual-22-12')
    query_embedding = np.array(response.embeddings[0], dtype=np.float32)

    # Search the Chroma collection for the top k results
    results = index.search(query_embedding, k_neighbors=10, complexity=200)

    # Format the results
    output = []
    for i in range(top_k):
        doc_id = results.identifiers[i]
        doc_title = docs[doc_id]['title']
        doc_text = docs[doc_id]['text']
        output.append({
            'id': doc_id,
            'title': doc_title,
            'text': doc_text
        })

    return output

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)