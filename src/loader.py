import json

def load_documents(path="dataset/raw/squad_subset.json"):
    """
    Loads the SQuAD JSON and returns de-duplicated documents
    Each document = {id, text} where id is a stable hash of the text.
    """
    
    documents = {} # keyed by context tet to dedupe
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            context = row["context"].strip()
            if context not in documents:
                doc_id = f"doc_{len(documents)}"
                documents[context] = doc_id
                
    # invert into list of {id, text}
    doc_list = [{"id": doc_id, "text": text} for text, doc_id in documents.items()]
    return doc_list

if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} unique documents")
    print("Sample doc:", docs[0])