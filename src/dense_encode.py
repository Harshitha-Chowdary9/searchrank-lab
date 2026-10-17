"""Standalone dense-embedding encoder (run once, before src.train).

Kept in its own process on purpose: sentence-transformers + torch have a
large memory footprint, and on small machines that collides with the BM25
index. This script encodes every product and query with
sentence-transformers/all-MiniLM-L6-v2 and saves the full query-product
cosine-similarity matrix (float16) that src.train.py then consumes without
loading transformers.
"""
import argparse

import numpy as np

from .data import download_data, load_products, load_queries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results/dense_sim.npz")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    products = load_products(args.data_dir)
    queries = load_queries(args.data_dir)
    pids = sorted(products.keys())
    qids = sorted(queries.keys())
    texts = [(products[pid]["name"] + ". " + products[pid]["description"][:220]).strip() for pid in pids]

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print(f"encoding {len(texts)} products ...", flush=True)
    emb = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True,
                       convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    print(f"encoding {len(qids)} queries ...", flush=True)
    q_emb = model.encode([queries[q] for q in qids], convert_to_numpy=True,
                         normalize_embeddings=True).astype(np.float32)
    sim = (q_emb @ emb.T).astype(np.float16)  # 480 x 42,994
    np.savez_compressed(args.out, sim=sim, qids=np.array(qids), pids=np.array(pids))
    print(f"wrote {args.out} shape {sim.shape}", flush=True)


if __name__ == "__main__":
    main()
