from sentence_transformers import SentenceTransformer
import yaml

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

class Embedder:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts):
        # Embed chunks for storage. No prefix - documents are encoded as is.
        return self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    
    def embed_query(self, text):
        # Embed a search query. BGE model expects a specific instruction prefix.
        prefixed = BGE_QUERY_PREFIX + text
        return self.model.encode(prefixed, normalize_embeddings=True)
    
if __name__ == "__main__":
    with open("../config.yaml") as f:
        config = yaml.safe_load(f)
        
    embedder = Embedder(model_name=config["embedding"]["model_name"])
    
    sample_chunks = ["The Eiffel Tower is located in Paris.", "Python is a programming language."]
    doc_embeddings = embedder.embed_documents(sample_chunks)
    query_embedding = embedder.embed_query("Where is the Eiffel Tower?")
    
    print(f"Document embeddings shape: {doc_embeddings.shape}")
    print(f"Query embedding shape: {query_embedding.shape}")