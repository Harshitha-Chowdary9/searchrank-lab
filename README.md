# searchrank-lab

Learning-to-rank for product search on the public **WANDS** dataset (Wayfair
ANnotation Dataset, ECIR 2022): 42,994 products, 480 real search queries,
231,873 human relevance judgements (Exact / Partial / Irrelevant).

> **Learning project.** Built by Harshitha Chittineni with AI-assistant
> pairing as a hands-on study project for search/data-science skills:
> information retrieval, semantic search, learning-to-rank, classical ML,
> PyTorch, systematic model evaluation, FastAPI and Docker. It is not
> production software.

## What it does

- **Retrieval, two ways, measured side by side** (full 42,994-product corpus):
  - **BM25 written from scratch** with an inverted index (`src/bm25.py`), and
  - **dense semantic search** with `all-MiniLM-L6-v2` embeddings, plus a
    min-max-normalised **hybrid** that fuses both score scales.
- **Feature engineering** for every judged (query, product) pair: BM25 score,
  TF-IDF cosine, name/description term overlap, exact-phrase flag, text
  statistics and product-quality signals (`src/features.py`).
- **Three rerankers trained on the same split**:
  - logistic regression (pointwise, scikit-learn),
  - **LightGBM LambdaMART** (gradient boosting with a listwise ranking
    objective),
  - **RankNet-style MLP in PyTorch** (pairwise logistic loss).
- **Systematic evaluation**: query-level 60/20/20 split (no query leaks into
  test), NDCG@5/@10, MAP, MRR, **bootstrap 95% confidence intervals**, a
  **paired bootstrap significance test** vs the BM25 baseline, and LightGBM
  gain-based feature importance - all in `results/`.
- **FastAPI search service** (`src/api.py`): BM25 candidates -> LambdaMART
  reranking, with `/health`, `/search` and `/metrics`, plus a `Dockerfile`
  for a self-contained runtime image.
- **LLM-answer evaluation harness** (`src/llm_eval/`): token F1, ROUGE-L, a
  rubric judge interface (deterministic offline default; optional
  OpenAI-compatible judge), and judge-vs-human agreement (percent agreement,
  Cohen's kappa) with bootstrap CIs.

## Measured results

### Reranking (96 held-out test queries, judged candidate set)

| ranker | NDCG@5 | NDCG@10 | MAP | MRR |
|---|---|---|---|---|
| BM25 (baseline) | 0.7561 | 0.7633 | 0.8746 | 0.9587 |
| LogReg (pointwise) | 0.7305 | 0.7408 | 0.8795 | 0.9607 |
| LambdaMART (LightGBM) | 0.7628 | 0.7657 | 0.8879 | 0.9647 |
| RankNet (PyTorch MLP) | 0.7463 | 0.7540 | 0.8766 | 0.9600 |

95% bootstrap CIs and the paired-bootstrap significance check vs BM25 are in `results/metrics.json`; the same numbers are plotted in `results/ranking_metrics.png`.

Paired bootstrap on NDCG@10 vs BM25 - LogReg (pointwise): mean diff -0.0225, CI [-0.0457, -0.0016] (significant); LambdaMART (LightGBM): mean diff +0.0024, CI [-0.0236, 0.0287] (not significant); RankNet (PyTorch MLP): mean diff -0.0093, CI [-0.0281, 0.009] (not significant).

### Full-corpus retrieval (Recall@k, test queries)

| method | recall@1 | recall@5 | recall@10 | recall@20 | recall@50 | recall@100 |
|---|---|---|---|---|---|---|
| bm25 | 0.0065 | 0.0303 | 0.0568 | 0.1068 | 0.2093 | 0.3262 |
| dense | 0.0073 | 0.0334 | 0.0625 | 0.1162 | 0.2248 | 0.3417 |
| hybrid | 0.0075 | 0.0325 | 0.0610 | 0.1188 | 0.2401 | 0.3709 |

Curves: `results/retrieval_recall.png`. Feature importance: `results/feature_importance.png`.

### LLM-answer evaluation harness (bundled mock outputs)

- `mock_assistant_v1`: token F1 0.701, ROUGE-L 0.635, judge agreement 0.75 (kappa 0.00)
- `mock_assistant_v1_noisy`: token F1 0.516, ROUGE-L 0.501, judge agreement 0.62 (kappa 0.41)


Numbers are reproducible with `python -m src.train` (seed fixed, 42).

## Run it

```bash
pip install -r requirements.txt          # CPU-only torch
python -m src.data                        # no-op helper; train downloads WANDS itself
python -m src.dense_encode                # one-off: dense similarity cache (~15 min CPU)
python -m src.train                       # full pipeline -> results/ + artifacts/
python -m src.llm_eval.run_eval           # LLM-eval harness -> results/llm_eval.json
python -m pytest                          # test suite
```

### API

```bash
uvicorn src.api:app --port 8000
curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "modern leather sofa", "k": 5}'
```

### Docker

```bash
docker build -t searchrank-lab .
docker run -p 8000:8000 searchrank-lab
```

The image downloads the public WANDS files at build time and serves the API
with the committed LambdaMART artifact.

## What this project deliberately does not claim

- The ranking evaluation **reranks the judged candidate set** per query
  (the standard WANDS setup - unjudged documents have no grade to score).
  Retrieval quality on the full corpus is reported separately as Recall@k.
- The dense encoder ran on CPU; numbers reflect that pipeline, not a tuned
  production system.
- The LLM-eval sample outputs are bundled mock assistant answers, clearly
  labelled - the harness (metrics, judge, agreement analysis) is the
  deliverable, not the samples. No paid LLM API was called.
- Not included: click-log labels, ANN vector indexing, online A/B testing,
  feature stores, monitoring. See `STUDY_GUIDE.md` for the interview-level
  reasoning behind each design choice.

## Dataset

WANDS: Chen et al., *WANDS: Dataset for Product Search Relevance
Assessment*, ECIR 2022. https://github.com/wayfair/WANDS
