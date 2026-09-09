from groq import Groq
from dotenv import load_dotenv
import yaml
import os

load_dotenv()

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided context.

Rules:
- If the answer is not contained in the context, say "I don't have enough information to answer that."
- Do not use any outside knowledge, even if you know the asnwer.
- Cite which piece of context you used by referencing its source_doc_id.
- Keep answers concise and directly grounded in the text.
"""

with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)
    

class Generator:
    def __init__(self, model_name=config["llm"]["model_name"]):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model_name = model_name
        
    def build_prompt(self, query, retrieved_chunks):
        context_blocks = []
        for chunk in retrieved_chunks:
            source_id = chunk["metadata"]["source_doc_id"]
            context_blocks.append(f"[source_doc_id: {source_id}]\n{chunk['text']}")
            
        context_text = "\n\n---\n\n".join(context_blocks)
        
        user_prompt = f"""Context:
        {context_text}
        
        Question: {query}
        
        Answer based only on the context above."""
        return user_prompt
    
    def generate(self, query, retrieved_chunks):
        user_prompt = self.build_prompt(query, retrieved_chunks)
        
        response = self.client.chat.completions.create(
            model = self.model_name,
            messages=[
                { "role": "system", "content": SYSTEM_PROMPT },
                { "role": "user", "content": user_prompt }
            ],
            temperature=0.1 # low temperature - factual grounding matters more than creativity here
        )
        return response.choices[0].message.content

if __name__=="__main__":
    # Minimal test with fake retrieved chunks, before wiring to the real retriever
    fake_chunks = [
        { "text": "The Eiffel Tower was completed in 1889.", "metadata": {"source_doc_id": "doc_0"} },
        { "text": "It was built as the entrance arch for the 1889 World's Fair.", "metadata": {"source_doc_id": "doc_0"} }
    ]
    
    generator = Generator()
    answer = generator.generate("When was the Eiffel Tower completed?", fake_chunks)
    print(answer)