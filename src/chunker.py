import re
import yaml

def split_sentences(text):
    sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    sentences = sentence_endings.split(text.strip())
    
    return [s.strip() for s in sentences if s.strip()]

def chunk_text(text, chunk_size=512, chunk_overlap=50):
    """
    Groups sentences into chunks up to chunk_size characters.
    Carriers trailing sentences into the next chunk to create overlap.
    """
    sentences = split_sentences(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        # if adding this sentence would exceed chunk_size, close the chunk
        if current_length + sentence_length > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            
            # build overlap: carry sentences from the end until we hit ~chunk_overlap
            overlap_sentences = []
            overlap_length = 0
            for s in reversed(current_chunk):
                if overlap_length + len(s) > chunk_overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_length += len(s)
            
            current_chunk = overlap_sentences
            current_length = overlap_length
        
        current_chunk.append(sentence)
        current_length += sentence_length
        
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from loader import load_documents
    
    with open("../config.yaml") as f:
        config = yaml.safe_load(f)
        
    docs = load_documents("../dataset/raw/squad_subset.json")
    sample = docs[5]["text"]
    
    chunks = chunk_text(
        sample,
        chunk_size=config["chunking"]["chunk_size"],
        chunk_overlap=config["chunking"]["chunk_overlap"]
    )
    
    print(f"Original length: {len(sample)} chars")
    print(f"Number of chunks: {len(chunks)}\n")
    
    for i, c in enumerate(chunks):
        print(f"[Chunk {i}] ({len(c)} chars): {c}\n")
    