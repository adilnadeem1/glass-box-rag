import sys
sys.path.append("../src")
from embedder import Embedder, BGE_QUERY_PREFIX
import numpy as np
import yaml

def test_embedder():
    with open("../config.yaml") as f:
        config = yaml.safe_load(f)
        
    embedder = Embedder(model_name=config["embedding"]["model_name"])
    
    # 1. Shape check - documents
    docs = ["Paris is the capital of France.", "The Nile is a river in Africa."]
    doc_embeddings = embedder.embed_documents(docs)
    assert doc_embeddings.shape == (2, 384), f"Expected (2, 384), got {doc_embeddings.shape}"
    print("Document Embedding shape correct")
    
    # 2. Shape check single query
    query_embeddings = embedder.embed_query("What is the capital of France?")
    assert query_embeddings.shape == (384,), f"Expected (384,), got {query_embeddings.shape}"
    print("Query embedding shapre correct")
    
    # 3. Normalization check - Vectors should have unit length (L2 norm ~1 )
    norm = np.linalg.norm(doc_embeddings[0])
    assert abs(norm - 1.0) < 1e-5, f"Expected normalized vector (norm=1), got norm={norm}"
    print("Embeddings are normalized")
    
    # 4. Determinism check - same input should give same output
    repeat_embedding = embedder.embed_query("What is the capital of France?")
    assert np.allclose(query_embeddings, repeat_embedding), "Same query gave different embeddings!"
    print("Embedding is deterministic")
    
    # 5. Query prefix is actually being applied (not silently skipped)
    raw_embedding = embedder.model.encode("test query", normalize_embeddings=True)
    prefix_embedding = embedder.embed_query("test query")
    assert not np.allclose(raw_embedding, prefix_embedding), "Query embedding matches raw text embedding - prefix isn't being applied"
    print("Query Prefix is being applied correctly")
    
    print("\n All embedder tests passed.")
    
if __name__ == "__main__":
    test_embedder()    