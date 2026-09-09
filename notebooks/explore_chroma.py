import chromadb

# In-memory client for this quick test (no persistence yet)
client = chromadb.Client()
collection = client.create_collection(
    name="test_collection",
    metadata={"hnsw:space": "cosine"}
)

# Chroma can embed automatically, but for the sake of control over embeddings
# (using embedder.py) the vectors are passed directly
collection.add(
    ids=["1", "2", "3"],
    documents=["Paris is the capital of France.", "The Nile is a river in Africa.", "Python is a programming language."],
    embeddings=[[0.1, 0.2, 0.3], [0.4, 0.1, 0.2], [0.9, 0.8, 0.1]], # fake vectors, just to see the API
    metadatas=[{"source": "doc_a"}, {"source": "doc_b"}, {"source": "doc_c"}]
)

results = collection.query(
    query_embeddings=[[0.1, 0.2, 0.31]], # close to vector 1
    n_results=2
)

print("Query results:")
print(results)