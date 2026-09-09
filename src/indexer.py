import chromadb
import yaml
import sys
from chunker import chunk_text
from loader import load_documents
from embedder import Embedder

class Indexer:
    def __init__(self, persist_directory, collection_name, embedder: Embedder):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.embedder = embedder
        
    def index_documents(self, documents, chunk_size, chunk_overlap):
        """
        documents: list of {"id": ..., "text": ...} from loader.py
        Chunks each document, embeds the chunks, and stores them in Chroma.
        """
        all_chunks = []
        all_ids = []
        all_metadatas = []
        
        for doc in documents:
            chunks = chunk_text(doc["text"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_ids.append(f"{doc['id']}_chunk_{i}")
                all_metadatas.append({"source_doc_id": doc["id"], "chunk_index": i})
        
        print(f"Embedding {len(all_chunks)} chunks from {len(documents)} documents...")
        embeddings = self.embedder.embed_documents(all_chunks)
        
        self.collection.add(
            ids=all_ids,
            documents=all_chunks,
            embeddings=embeddings.tolist(), # Chroma expects plain lists, not numpy arrays
            metadatas=all_metadatas 
        )
        print(f"Indexed {len(all_chunks)} chunks into collection '{self.collection.name}'")
        
    def count(self):
        return self.collection.count()
    
if __name__ == "__main__":
    with open("../config.yaml") as f:
        config = yaml.safe_load(f)
        
    embedder = Embedder(model_name=config["embedding"]["model_name"])
    indexer = Indexer(
        persist_directory=f"../{config['vector_store']['persist_directory']}",
        collection_name=config["vector_store"]["collection_name"],
        embedder=embedder
    )
    
    docs = load_documents("../dataset/raw/squad_subset.json")
    indexer.index_documents(
        docs,
        chunk_size=config["chunking"]["chunk_size"],
        chunk_overlap=config["chunking"]["chunk_overlap"]
    )
    
    print(f"Total chunks in collection: {indexer.count()}")
        