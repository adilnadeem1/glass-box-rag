import sys
sys.path.append("../src")
import json
from pipeline import RAGPipeline

# Pull a handful of real question/answer pairs from raw data
qa_samples = []
with open("../dataset/raw/squad_subset.json") as f:
    for line in f:
        row = json.loads(line)
        if row["answers"]["text"]:
            qa_samples.append({
                "question": row["question"],
                "ground_truth": row["answers"]["text"][0]
            })
            
pipeline = RAGPipeline()

# Test on a handful, not all - this is a manual eyeball check, not automated eval yet
for sample in qa_samples[:5]:
    result = pipeline.run(sample["question"], verbose=False)
    print(f"Q: {sample['question']}")
    print(f"Ground truth: {sample['ground_truth']}")
    print(f"Generated: {result['answer']}")
    print("-" * 80)