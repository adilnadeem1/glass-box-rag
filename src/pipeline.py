import yaml
from dotenv import load_dotenv
from embedder import Embedder
from retriever import Retriever
from generator import Generator

load_dotenv()

class RAGPipeline:
    def __init__(self, config_path="../config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.embedder = Embedder(model_name=self.config["embedding"]["model_name"])
        self.retriever = Retriever(
            persist_directory=f"../{self.config['vector_store']['persist_directory']}",
            collection_name=self.config["vector_store"]["collection_name"],
            embedder = self.embedder
        )
        
        self.generator = Generator()
        
    def run(self, query: str, top_k: int = 3, verbose: bool = True):
        retrieved_chunks = self.retriever.retrieve(query, top_k=top_k)
        
        if verbose:
            print(f"\n--- Retrieved {len(retrieved_chunks)} chunks ---")
            for chunk in retrieved_chunks:
                print(f"[distance={chunk['distance']:.4f}] {chunk['text'][:100]}...")
                
        answer = self.generator.generate(query, retrieved_chunks)
        
        return {
            "query": query,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks
        }
        
if __name__ == "__main__":
    pipeline = RAGPipeline()
    
    query = "What year was the university founded?" # to be swapped with a SQuAD related question
    result = pipeline.run(query)
    
    print(f"\nQuery: {result['query']}")
    print(f"Answer: {result['answer']}")