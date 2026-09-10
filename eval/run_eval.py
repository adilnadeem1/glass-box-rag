import sys
sys.path.append("../src")
import json
from pipeline import RAGPipeline
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_groq import ChatGroq
import os
import yaml
from dotenv import load_dotenv

load_dotenv()

def run_pipeline_over_eval_set(eval_set_path="eval_set.json"):
    with open(eval_set_path) as f:
        eval_samples = json.load(f)
        
    pipeline = RAGPipeline()
    
    results = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    for sample in eval_samples:
        output = pipeline.run(sample["question"], top_k=3, verbose=False)
        
        results["question"].append(sample["question"])
        results["answer"].append(output["answer"])
        results["contexts"].append([c["text"] for c in output["retrieved_chunks"]])
        results["ground_truth"].append(sample["ground_truth"])
        
    return  Dataset.from_dict(results)

if __name__=="__main__":
    with open("../config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    dataset = run_pipeline_over_eval_set()
    
    # RAGAS needs an LLM judge - reuse Groq instead of defaulting to OpenAI
    judge_llm = LangchainLLMWrapper(ChatGroq(
        model=config["llm"]["model_name"],
        api_key=os.environ.get("GROQ_API_KEY")
    ))
    
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=config["embedding"]["model_name"])
    )
    
    answer_relevancy.strictness = 1 # Groq's API rejects n>1; disable self-consistency sampling
    
    run_config = RunConfig(max_workers=2, timeout=120)
    
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config
    )
    
    print(scores)
    scores.to_pandas().to_csv("eval_results.csv", index=False)
    print("Saved detailed results to eval_results.csv")