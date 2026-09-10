# RAG POC

A from-scratch Retrieval-Augmented Generation pipeline, built step by step to
understand each block before reaching for a framework.

## Stack (free tier / local)
- **Documents**: SQuAD 2.0 subset (Hugging Face `datasets`)
- **Embeddings**: `BAAI/bge-small-en-v1.5` (local, via `sentence-transformers`)
- **Vector store**: Chroma (embedded, local)
- **LLM**: Groq free tier (Llama 3.1)
- **Eval**: RAGAS

## Setup

```bash
conda create -n rag-poc python=3.11 -y
conda activate rag-poc
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your API key(s).

## Project structure
```
rag-poc/
├── dataset/
│   ├── raw/                          # Original documents, untouched
│   │   └── squad_subset.json
│   └── processed/                    # Chunked/cleaned output, vector DB persistence
│   │   └── chroma_db/                # Chroma's persistent index (SQLite + HNSW files)
│
├── src/                              # Pipeline code
│   ├── download_data.py              # Pulls SQuAD 2.0 subset from Hugging Face
│   ├── loader.py                     # Loads and de-duplicates documents
│   ├── chunker.py                    # Sentence-aware chunking with overlap
│   ├── embedder.py                   # Wraps bge-small-en-v1.5 and handles query/doc prefix distinction
│   ├── indexer.py                    # chunks + embeds documents, persists to Chroma
│   ├── retriever.py                  # queries Chroma, returns top-k chunks for a query
│   ├── generator.py                  # builds grounding prompt, calls Groq LLM
│   └── pipeline.py                   # chains retriever + generator into one callable RAGPipeline
│
├── notebooks/                        # Exploration/debugging, not pipeline code
│   ├── explore_chunking.py           # Diagnostic: naive vs. sentence-aware chunking
│   ├── explore_embeddings.py         # Diagnostic: verifies semantic similarity behaves as expected
│   ├── explore_llm.py                # sanity check: raw Groq API call before RAG logic
│   └── explore_pipeline_quality.py   # manual eyeball check: generated answers vs SQuAD ground truth
│
├── eval/                             # Evaluation sets and evaluation scripts 
│                           
├── tests/                            # Testing scripts for different modules
│   └── test_embedder.py              # contract tests: shape, normalization, determinism, query-prefix behavior
│
├── config.yaml                       # Tunable pipeline parameters
├── .env                              # Secrets, gitignored
└── .env.example                      # Template for secrets
```


## Pipeline stages

| Stage | Script | Status |
|---|---|---|
| Config & env setup | `config.yaml`, `.env` | ✅ |
| Data download | `src/download_data.py` | ✅ |
| Document loading & dedup | `src/loader.py` | ✅ |
| Chunking | `src/chunker.py` | ✅ |
| Embedding | `src/embedder.py` | ✅ |
| Vector store indexing | `src/indexer.py` | ✅ |
| Retrieval | `src/retriever.py` | ✅ |
| Generation (LLM) | `src/generator.py` | ✅ |
| End-to-end pipeline | `src/pipeline.py` | ✅ |
| Evaluation | `eval/run_eval.py` | ⬜ |

## How to run what exists so far

```bash
# Download the SQuAD subset (500-row sample)
python src/download_data.py

# Load and de-duplicate documents -> 43 unique paragraphs
python src/loader.py

# Chunk a sample document (sentence-aware, with overlap)
cd src
python chunker.py
```


## Testing

Lightweight contract tests (no pytest yet — plain assert scripts) live alongside
each module as `test_<module>.py`. Run individually as each stage is built:

```bash
cd tests
python test_embedder.py
```

## Data notes

- Sampling `train[:500]` from SQuAD 2.0 yields **43 unique documents** after
  de-duplication (SQuAD has multiple Q&A pairs per paragraph). Sufficient for a
  POC — small enough to manually inspect every chunk, large enough to see
  chunking/retrieval behavior. Increase the slice size in `download_data.py`
  if more document variety is needed later.

## Design notes

- **Why de-dupe documents?** SQuAD has multiple Q&A pairs per paragraph —
  indexing the same paragraph repeatedly would skew retrieval toward
  duplicated content.
- **Why a `config.yaml`?** Chunk size, overlap, and model names get tuned
  repeatedly during retrieval-quality iteration — centralizing them avoids
  hunting through code for hardcoded values.
- **Why sentence-aware chunking over fixed-size?** Fixed-size character
  splitting cuts mid-sentence/mid-word (verified on real SQuAD paragraphs in
  `notebooks/explore_chunking.py`), producing embeddings for incomplete
  thoughts that don't cluster near their true semantic meaning. Grouping by
  sentence boundaries preserves complete ideas per chunk. Overlap (carrying
  trailing sentences into the next chunk) prevents context loss when a
  concept spans a chunk boundary.
- **Why `bge-small-en-v1.5`?** Free, runs locally (no API cost/rate limits),
  384-dim vectors keep the vector store small for a POC, and it ranks well on
  the MTEB retrieval benchmark for its size class.
- **Query vs. document encoding asymmetry**: BGE models require queries to be
  prefixed with an instruction string ("Represent this sentence for searching
  relevant passages: ") but documents are encoded as-is. Skipping this prefix
  on queries silently degrades retrieval quality without throwing an error -
  verified via `notebooks/explore_embeddings.py` diagnostic (unrelated vs.
  relevant query similarity comparison).
- **Why `normalize_embeddings=True`?** L2-normalizes vectors so cosine
  similarity reduces to a dot product - cheaper at scale, and matches what
  Chroma assumes by default.
- **Why test the query-prefix behavior explicitly?** It's the kind of bug that
  fails silently - if `embed_query` accidentally stopped applying the BGE
  instruction prefix, embeddings would still have the right shape and still
  work, just retrieve worse. A shape-only test wouldn't catch this; an explicit
  "prefixed embedding != raw embedding" assertion does.
- **Why Chroma?** Embedded (in-process, no separate server), persists to disk
  with minimal setup - fits a laptop-based POC. Swappable for FAISS or a
  hosted store later without changing the rest of the pipeline, since indexing
  and retrieval are isolated behind `Indexer`/`Retriever` classes.
- **Why explicit `hnsw:space: cosine`?** Embeddings are L2-normalized
  (see embedder.py), so cosine distance is the correct metric. Chroma
  defaults to L2 distance, which is rank-equivalent for normalized vectors
  but produces different distance values - matters once similarity
  thresholds are introduced.
- **Why cosine distance means "smaller = more similar"?** Distance, not
  similarity - Chroma returns `1 - cosine_similarity` under the hood when
  configured for cosine space, so lower values indicate closer/better matches.
- **Why a strict grounding system prompt?** Without explicit instructions to
  refuse answering outside the given context, LLMs default to blending
  retrieved context with their own training knowledge - undermining the
  entire point of RAG (traceable, verifiable answers). The prompt explicitly
  forbids outside knowledge and requires a fallback ("I don't have enough
  information") rather than guessing.
- **Why low temperature (0.1) for generation?** Factual grounding is
  prioritized over creative phrasing - lower temperature reduces the
  likelihood of the model drifting from the literal provided context.
- **Why test generation with fake chunks first?** Isolates whether a bug
  lives in the prompt/LLM call itself vs. in how retrieved chunks are
  formatted and passed in from `retriever.py`.
- **Why build components once in `__init__`, not per-query?** Embedder model
  loading and the Chroma client connection are expensive one-time costs.
  Separating setup (`__init__`) from per-query work (`run`) avoids repeating
  that cost on every question - matters once running batch evaluation
  (Step 9) over many queries.
- **Why manually eyeball answers before automated eval?** A quick pass over
  5-10 real question/ground-truth pairs surfaces obvious systemic issues
  (bad retrieval vs. bad generation) faster than running a full metric suite
  blind. It also tells you which failure mode to expect before interpreting
  RAGAS scores in Step 9 - a low faithfulness score means something
  different if you already know retrieval is solid vs. shaky.

## Known issues / gotchas

- Bare `squad_v2` dataset ID works via `datasets.load_dataset` currently, but
  some `huggingface_hub` versions require the namespaced form
  `rajpurkar/squad_v2` — use that if you hit a Hub URI validation error.
- Sentence splitter in `chunker.py` is regex-based (not a full NLP model) —
  may over/under-split on edge cases like abbreviations mid-sentence.
- First run of `embedder.py` downloads the model (~130MB) from Hugging Face -
  expect a delay on first execution only (cached afterward in
  `~/.cache/huggingface`).
- Chroma's `.add()` requires plain Python lists for embeddings, not numpy
  arrays - `sentence-transformers` returns numpy, so `.tolist()` conversion
  is required in `indexer.py`.
- Chroma query results are structured as lists-of-lists (one outer list per
  query in the batch) - `results["documents"][0]` is the actual result list,
  not `results["documents"]`
- SQuAD 2.0 includes unanswerable questions (`answers.text` is empty) -
  filter these out for initial pipeline sanity checks; they become useful
  once specifically testing whether the system correctly refuses to answer
  (see eval stage)