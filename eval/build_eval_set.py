import json

def build_eval_set(raw_path, output_path, n_samples=20):
    """
    Pulls answerable questions from the SQuAD raw file to build a small,
    fixed eval set. Fixed size and saved to disk so eval runs are 
    reproducible and comparable across pipeline changes.
    """
    eval_samples = []
    with open(raw_path) as f:
        for line in f:
            row = json.loads(line)
            if row["answers"]["text"]: # skip answerable for the core eval set
                eval_samples.append({
                    "question": row["question"],
                    "ground_truth": row["answers"]["text"][0]
                })
            if len(eval_samples) >= n_samples:
                break
    
    with open(output_path, "w") as f:
        json.dump(eval_samples, f, indent=2)
        
    print(f"Built eval set with {len(eval_samples)} samples -> {output_path}")
    return eval_samples

if __name__=="__main__":
    build_eval_set(
        raw_path="../dataset/raw/squad_subset.json",
        output_path="eval_set.json",
        n_samples=20
    )
    