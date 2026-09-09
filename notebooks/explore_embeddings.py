import sys
sys.path.append("../src")
from sentence_transformers import SentenceTransformer
from chunker import chunk_text
from loader import load_documents
from sklearn.metrics.pairwise import cosine_similarity

# First Run will download the model(~130MB) from Hugging Face - one-time cost
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

docs = load_documents("../dataset/raw/squad_subset.json")
sample_text = docs[5]["text"]
chunks = chunk_text(sample_text, chunk_size=512, chunk_overlap=50)

embeddings = model.encode(chunks)

print(f"Number of chunks: {len(chunks)}")
print(f"Embedding shape: {embeddings.shape}") # (num_chunks, 384) for bge-small
print(f"First embedding vector (first 10 dims): {embeddings[0][:10]}")

# Two questions - one relevant to the sample doc, one about something unrelated
query_relevant = "What is this passage about?"
query_unrelated = "How do you bake a choclate cake?"

query_embeddings = model.encode([query_relevant, query_unrelated])

similarities_relevant = cosine_similarity([query_embeddings[0]], embeddings)[0]
similarities_unrelated = cosine_similarity([query_embeddings[1]], embeddings)[0]

print(f"\n Avg Similarity (relevant-ish query): {similarities_relevant.mean():.4f}")
print(f"Avg similarity (unrelated query): {similarities_unrelated.mean():.4f}")