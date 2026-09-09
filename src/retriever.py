import chromadb
import yaml
from embedder import Embedder

class Retriever:
    def __init__(self, persist_directory, collection_name, embedder: Embedder):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_collection(name=self.collection_name)
        self.embedder = embedder
        
    def retrieve(self, query: str, top_k: int = 5):
        query_embedding = self.embedder.embed_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        # flatten Chroma's list of lists response into a simple list of dicts
        retrieved = []
        for i in range(len(results["ids"][0])):
            retrieved.append({
                "id"        : results["ids"][0][i],
                "text"      : results["documents"][0][i],
                "metadata"  : results["metadatas"][0][i],
                "distance"  : results["distances"][0][i]
            })
        return retrieved
    
if __name__ == "__main__":
    with open("../config.yaml") as f:
        config = yaml.safe_load(f)
        
    embedder = Embedder(model_name=config["embedding"]["model_name"])
    retriever = Retriever(
        persist_directory= f"../{config['vector_store']['persist_directory']}",
        collection_name=config["vector_store"]["collection_name"],
        embedder=embedder
    )
    
    query = "What year was the university founded?"
    results = retriever.retrieve(query, top_k=3)
    
    print(f"Query: {query}\n")
    for r in results:
        print(f"[distance={r['distance']:.4f}] {r['text'][:150]}...\n")