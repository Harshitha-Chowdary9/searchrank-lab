"""FastAPI search service: BM25 retrieval + LambdaMART reranking.

Endpoints:
  GET  /health   - liveness + corpus size
  POST /search   - {"query": "...", "k": 10, "candidates": 50} -> ranked products
  GET  /metrics  - request counts and latency

Startup rebuilds BM25 and TF-IDF from data/product.csv (a few seconds) and
loads the trained LambdaMART model from artifacts/. Set SEARCHRANK_DATA_DIR
and SEARCHRANK_ARTIFACTS_DIR to override the defaults.
"""
import os
import time

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .bm25 import BM25
from .data import load_products
from .features import FEATURE_NAMES, assemble_row, text_overlap_features
from .text import tokenize

DATA_DIR = os.environ.get("SEARCHRANK_DATA_DIR", "data")
ARTIFACTS_DIR = os.environ.get("SEARCHRANK_ARTIFACTS_DIR", "artifacts")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    k: int = Field(10, ge=1, le=50)
    candidates: int = Field(50, ge=1, le=200)


def create_app(data_dir=DATA_DIR, artifacts_dir=ARTIFACTS_DIR):
    import json

    import lightgbm as lgb
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    products = load_products(data_dir)
    pids = sorted(products.keys())
    docs = [(products[pid]["name"] + " " + products[pid]["description"]).strip() for pid in pids]
    bm25 = BM25([tokenize(d) for d in docs])
    tfidf = TfidfVectorizer(lowercase=True, token_pattern=r"[a-z0-9]+", max_features=60000)
    doc_matrix = tfidf.fit_transform(docs)
    with open(os.path.join(artifacts_dir, "feature_config.json")) as fh:
        config = json.load(fh)
    booster = lgb.Booster(model_file=os.path.join(artifacts_dir, "lgbm_ranker.txt"))
    assert config["feature_names"] == FEATURE_NAMES, "artifact/feature mismatch"

    app = FastAPI(title="searchrank-lab", version="0.1.0")
    state = {"requests": 0, "searches": 0, "total_latency_ms": 0.0}

    @app.get("/health")
    def health():
        return {"status": "ok", "products": len(pids)}

    @app.post("/search")
    def search(req: SearchRequest):
        t0 = time.perf_counter()
        candidates = bm25.top_n(tokenize(req.query), n=req.candidates)
        if not candidates:
            state["requests"] += 1
            return {"query": req.query, "results": []}
        q_vec = tfidf.transform([req.query])
        rows = []
        for doc_id, bm25_score in candidates:
            pid = pids[doc_id]
            product = products[pid]
            base = text_overlap_features(req.query, product)
            from sklearn.metrics.pairwise import cosine_similarity
            cos = float(cosine_similarity(q_vec, doc_matrix[doc_id])[0, 0])
            rows.append((pid, assemble_row(base, product, bm25_score, cos)))
        scores = booster.predict(np.array([r[1] for r in rows], dtype=np.float32))
        order = sorted(range(len(rows)), key=lambda i: -scores[i])[: req.k]
        latency = (time.perf_counter() - t0) * 1000
        state["requests"] += 1
        state["searches"] += 1
        state["total_latency_ms"] += latency
        return {
            "query": req.query,
            "latency_ms": round(latency, 1),
            "results": [
                {
                    "product_id": rows[i][0],
                    "name": products[rows[i][0]]["name"],
                    "class": products[rows[i][0]]["class"],
                    "score": round(float(scores[i]), 4),
                }
                for i in order
            ],
        }

    @app.get("/metrics")
    def metrics():
        avg = state["total_latency_ms"] / state["searches"] if state["searches"] else 0.0
        return {"requests": state["requests"], "searches": state["searches"],
                "avg_search_latency_ms": round(avg, 1)}

    return app


app = create_app() if os.environ.get("SEARCHRANK_SKIP_AUTOSTART") != "1" else None
