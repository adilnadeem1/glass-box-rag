from datasets import load_dataset
import json
import os

def download_squad(output_dir="dataset/raw"):
    os.makedirs(output_dir, exist_ok=True)
    
    print('Downloading SQuAD 2.0...')
    dataset = load_dataset("rajpurkar/squad_v2", split="train[:500]")  # small slice for POC
    
    output_path = os.path.join(output_dir, "squad_subset.json")
    dataset.to_json(output_path)
    
    print(f"Saved {len(dataset)} examples to {output_path}")
    return output_path

if __name__ == "__main__":
    download_squad()