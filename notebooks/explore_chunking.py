import sys
sys.path.append("../src")
from loader import load_documents

docs = load_documents("../dataset/raw/squad_subset.json")

# Pick one document and look at it
sample = docs[5]["text"]
print("Full paragraph:")
print(sample)
print(f"\nLength: {len(sample)} characters")

# Naive fixed-size split, no awareness of sentence boundaries
chunk_size = 200
naive_chunks = [sample[i:i+chunk_size] for i in range(0, len(sample), chunk_size)]

print("\n --- Naive Chunks ---")
for i, chunk in enumerate(naive_chunks):
    print(f"[Chunk {i}]: {chunk}\n")