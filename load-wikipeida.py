# %%
from datasets import load_dataset
import diskannpy
import numpy as np
import os
from tqdm import tqdm

# %%
# Load the Wikipedia dataset with embeddings
docs = load_dataset("Cohere/wikipedia-22-12-en-embeddings", split="train", cache_dir='./datasets-local')

# Create a directory for the disk index
index_dir = "./wikipedia-index"
os.makedirs(index_dir, exist_ok=True)

# # Convert the dataset to a numpy array
# embeddings = np.array(docs['emb'], dtype=np.float32)

# # Build the disk index
# diskannpy.build_disk_index(
#     data=embeddings,
#     distance_metric="mips",
#     index_directory=index_dir,
#     complexity=196,
#     graph_degree=128,
#     search_memory_maximum=2,  # Adjust based on your system's memory
#     build_memory_maximum=64,
#     num_threads=32,  # Adjust based on your system's number of threads
# )


# %%
# Initialize the DynamicMemoryIndex
index = diskannpy.DynamicMemoryIndex(
    distance_metric="mips",
    vector_dtype=np.float32,
    dimensions=768,
    max_vectors=int(len(docs)*1.05),
    complexity=196,
    graph_degree=128,
    num_threads=32,  # Adjust based on your system's number of threads
)

print("Index initialized.")

# Build the index incrementally
for doc in tqdm(docs):
    embedding = np.array(doc['emb'], dtype=np.float32)
    index.insert(embedding, doc['id'])

print("Index built.")

index.save(index_dir, "wikipedia")

print("Wikipedia dataset indexed with diskannpy.")
print(f"Index saved to {index_dir}.")