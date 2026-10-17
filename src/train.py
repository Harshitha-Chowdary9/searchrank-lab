"""End-to-end training and evaluation pipeline.

Stages (each prints progress):
  1. load WANDS, split queries 60/20/20 (train/val/test, seeded)
  2. build BM25 + TF-IDF over the full 43k-product corpus
  3. retrieval evaluation on test queries (Recall@k: BM25 vs dense vs hybrid)
  4. feature engineering for every judged (query, product) pair
  5. train three rerankers: logistic regression (pointwise),
     LightGBM LambdaMART (listwise-ish, gradient boosting),
     RankNet MLP (pairwise, PyTorch)
  6. evaluation on held-out test queries: NDCG@5/@10, MAP, MRR with
     bootstrap 95% CIs and a paired bootstrap significance check vs BM25
  7. save artifacts (LambdaMART model + config) and results (JSON + plots)
"""
import argparse
import json
import os
import random
import time

import numpy as np

from .bm25 import BM25
from .data import download_data, load_labels, load_products, load_queries
from .features import FEATURE_NAMES, assemble_row, text_overlap_features
from .metrics import (
    bootstrap_ci,
    mean_metric,
    paired_bootstrap_diff,
    per_query_metrics,
)
from .text import tokenize


def split_queries(query_ids, seed=42, train=0.6, val=0.2):
    rng = random.Random(seed)
    ids = list(query_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(train * n)
    n_val = int(val * n)
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


def build_pairs(qids, queries, labels, products, bm25_maps, tfidf, doc_matrix, pid_index):
    """Feature matrix + grades + group sizes for a set of query ids."""
    X, y, group, meta = [], [], [], []
    for qid in qids:
        query = queries[qid]
        scores = bm25_maps[qid]
        from sklearn.metrics.pairwise import cosine_similarity  # local import, cheap
        q_vec = tfidf.transform([query])
        rows = []
        for pid, grade in labels[qid].items():
            product = products[pid]
            base = text_overlap_features(query, product)
            cos = float(cosine_similarity(q_vec, doc_matrix[pid_index[pid]])[0, 0])
            rows.append((assemble_row(base, product, scores.get(pid, 0.0), cos), grade, pid))
        for row, grade, pid in rows:
            X.append(row)
            y.append(grade)
            meta.append((qid, pid))
        group.append(len(rows))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), group, meta


def ranked_by_scores(meta, scores):
    """Turn flat (qid, pid) + flat scores into {qid: [pid, ...] best-first}."""
    per_q = {}
    for (qid, pid), s in zip(meta, scores):
        per_q.setdefault(qid, []).append((pid, float(s)))
    return {q: [pid for pid, _ in sorted(v, key=lambda kv: -kv[1])] for q, v in per_q.items()}


def evaluate_ranker(name, ranked_lists, labels, test_qids):
    pq = per_query_metrics({q: ranked_lists[q] for q in test_qids}, labels)
    out = {}
    for metric in ("ndcg@5", "ndcg@10", "map", "mrr"):
        vals = [m[metric] for q, m in pq.items() if m.get(metric) is not None]
        lo, hi = bootstrap_ci(vals)
        out[metric] = {"mean": round(mean_metric(pq, metric), 4),
                       "ci95": [round(lo, 4), round(hi, 4)], "n_queries": len(vals)}
    print(f"  {name:<22} NDCG@10 {out['ndcg@10']['mean']:.4f} "
          f"CI {out['ndcg@10']['ci95']}  MAP {out['map']['mean']:.4f}", flush=True)
    return out, pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--skip-dense", action="store_true", help="skip sentence-transformers retrieval")
    args = ap.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.artifacts_dir, exist_ok=True)
    t0 = time.time()

    print("[1/7] loading data", flush=True)
    download_data(args.data_dir)
    products = load_products(args.data_dir)
    queries = load_queries(args.data_dir)
    labels = load_labels(args.data_dir)
    print(f"  {len(products)} products, {len(queries)} queries, "
          f"{sum(len(v) for v in labels.values())} judgments", flush=True)
    train_q, val_q, test_q = split_queries(labels.keys())
    print(f"  split: {len(train_q)} train / {len(val_q)} val / {len(test_q)} test queries", flush=True)

    print("[2/7] building BM25 + TF-IDF over the full corpus", flush=True)
    pids = sorted(products.keys())
    pid_index = {pid: i for i, pid in enumerate(pids)}
    docs = [(products[pid]["name"] + " " + products[pid]["description"]).strip() for pid in pids]
    doc_tokens = [tokenize(d) for d in docs]
    bm25 = BM25(doc_tokens)
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(lowercase=True, token_pattern=r"[a-z0-9]+", max_features=60000)
    doc_matrix = tfidf.fit_transform(docs)
    print(f"  corpus indexed: {len(pids)} docs, tfidf vocab {len(tfidf.vocabulary_)}", flush=True)

    print("[3/7] retrieval evaluation (full-corpus Recall@k on test queries)", flush=True)
    ks = [1, 5, 10, 20, 50, 100]
    bm25_top = {}
    retrieval = {"bm25": {k: [] for k in ks}, "dense": {k: [] for k in ks}, "hybrid": {k: [] for k in ks}}
    dense_sim = None
    dense_qrow = {}
    if not args.skip_dense:
        import numpy as _np
        try:
            cache = _np.load(os.path.join(args.results_dir, "dense_sim.npz"))
            dense_sim = cache["sim"]
            cache_pids = cache["pids"].tolist()
            assert cache_pids == pids, "dense cache was built for a different product set"
            for row_i, qid in enumerate(cache["qids"].tolist()):
                dense_qrow[qid] = row_i
            print("  loaded precomputed dense similarity matrix", flush=True)
        except FileNotFoundError:
            print("  dense retrieval skipped: results/dense_sim.npz missing "
                  "(run python -m src.dense_encode first)", flush=True)
    for qid in test_q:
        query = queries[qid]
        bm = bm25.top_n(tokenize(query), n=100)
        bm25_top[qid] = [pids[i] for i, _ in bm]
        rel = {pid for pid, g in labels[qid].items() if g >= 1}
        if not rel:
            continue
        for k in ks:
            retrieval["bm25"][k].append(len(set(bm25_top[qid][:k]) & rel) / len(rel))
        if dense_sim is not None and qid in dense_qrow:
            sim_row = dense_sim[dense_qrow[qid]].astype(np.float32)
            dense_top = np.argpartition(-sim_row, 100)[:100]
            dense_rank = sorted(dense_top.tolist(), key=lambda i: -sim_row[i])
            # min-max normalise BM25 over its own top-100, fuse with dense cosine
            bm_scores = np.array([s for _, s in bm])
            lo_b, hi_b = float(bm_scores.min()), float(bm_scores.max())
            span = (hi_b - lo_b) or 1.0
            fused = {}
            cand = {pids[i] for i, _ in bm} | {pids[i] for i in dense_rank}
            bm_full = {pids[i]: s for i, s in bm}
            for pid in cand:
                fused[pid] = ((bm_full.get(pid, lo_b) - lo_b) / span) + float(sim_row[pid_index[pid]])
            hybrid_rank = sorted(fused.items(), key=lambda kv: -kv[1])[:100]
            for k in ks:
                retrieval["dense"][k].append(len({pids[i] for i in dense_rank[:k]} & rel) / len(rel))
                retrieval["hybrid"][k].append(len({p for p, _ in hybrid_rank[:k]} & rel) / len(rel))
    retrieval_out = {}
    for name, per_k in retrieval.items():
        if name != "bm25" and dense_sim is None:
            continue
        retrieval_out[name] = {f"recall@{k}": round(float(np.mean(v)), 4) for k, v in per_k.items() if v}
    print(f"  retrieval: {json.dumps(retrieval_out)}", flush=True)

    print("[4/7] feature engineering for judged pairs", flush=True)
    bm25_maps = {}
    for qid in labels:
        bm25_maps[qid] = {pids[i]: s for i, s in bm25.scores(tokenize(queries[qid])).items()}
    X_tr, y_tr, g_tr, meta_tr = build_pairs(train_q, queries, labels, products, bm25_maps, tfidf, doc_matrix, pid_index)
    X_va, y_va, g_va, meta_va = build_pairs(val_q, queries, labels, products, bm25_maps, tfidf, doc_matrix, pid_index)
    X_te, y_te, g_te, meta_te = build_pairs(test_q, queries, labels, products, bm25_maps, tfidf, doc_matrix, pid_index)
    print(f"  pairs: {len(X_tr)} train / {len(X_va)} val / {len(X_te)} test, {len(FEATURE_NAMES)} features", flush=True)

    print("[5/7] training rerankers", flush=True)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_tr)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(scaler.transform(X_tr), (y_tr >= 1).astype(int))
    print("  logistic regression (pointwise) done", flush=True)

    import lightgbm as lgb
    dtrain = lgb.Dataset(X_tr, label=y_tr, group=g_tr, feature_name=FEATURE_NAMES)
    dval = lgb.Dataset(X_va, label=y_va, group=g_va, reference=dtrain, feature_name=FEATURE_NAMES)
    params = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [5, 10],
              "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 50,
              "feature_fraction": 0.9, "verbosity": -1, "seed": 42}
    booster = lgb.train(params, dtrain, num_boost_round=500, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
    print(f"  LightGBM LambdaMART done (best iteration {booster.best_iteration})", flush=True)

    from .models.torch_ranker import predict_ranknet, train_ranknet
    ranknet, rn_mean, rn_std = train_ranknet(X_tr, y_tr, g_tr, epochs=8)

    print("[6/7] evaluation on held-out test queries", flush=True)
    results = {"dataset": "WANDS (Wayfair ANnotation Dataset, ECIR 2022)",
               "split": {"train_queries": len(train_q), "val_queries": len(val_q),
                         "test_queries": len(test_q), "seed": 42},
               "evaluation_note": "Reranking of the judged candidate set per query; "
                                  "unjudged documents count as grade 0. Retrieval Recall@k "
                                  "is measured separately over the full 42,994-product corpus.",
               "retrieval": retrieval_out}
    ranked = {}
    ranked["BM25 (baseline)"] = ranked_by_scores(meta_te, [bm25_maps[q].get(p, 0.0) for q, p in meta_te])
    ranked["LogReg (pointwise)"] = ranked_by_scores(meta_te, lr.predict_proba(scaler.transform(X_te))[:, 1])
    ranked["LambdaMART (LightGBM)"] = ranked_by_scores(meta_te, booster.predict(X_te))
    ranked["RankNet (PyTorch MLP)"] = ranked_by_scores(meta_te, predict_ranknet(ranknet, X_te, rn_mean, rn_std))
    results["ranking"] = {}
    per_query_store = {}
    for name, rl in ranked.items():
        out, pq = evaluate_ranker(name, rl, labels, test_q)
        results["ranking"][name] = out
        per_query_store[name] = pq
    # paired bootstrap vs BM25 on NDCG@10
    base_vals = [per_query_store["BM25 (baseline)"][q]["ndcg@10"] for q in test_q]
    results["paired_bootstrap_vs_bm25_ndcg10"] = {}
    for name in ranked:
        if name == "BM25 (baseline)":
            continue
        vals = [per_query_store[name][q]["ndcg@10"] for q in test_q]
        lo, hi = paired_bootstrap_diff(vals, base_vals)
        results["paired_bootstrap_vs_bm25_ndcg10"][name] = {
            "mean_diff": round(float(np.mean(vals) - np.mean(base_vals)), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "significant": bool(lo > 0 or hi < 0)}
    imp = booster.feature_importance(importance_type="gain")
    results["feature_importance_gain"] = {n: round(float(v), 1) for n, v in
                                          sorted(zip(FEATURE_NAMES, imp), key=lambda kv: -kv[1])}
    with open(os.path.join(args.results_dir, "metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("  wrote results/metrics.json", flush=True)

    print("[7/7] saving artifacts + plots", flush=True)
    booster.save_model(os.path.join(args.artifacts_dir, "lgbm_ranker.txt"))
    with open(os.path.join(args.artifacts_dir, "feature_config.json"), "w") as fh:
        json.dump({"feature_names": FEATURE_NAMES, "bm25_k1": 1.5, "bm25_b": 0.75,
                   "model": "LightGBM LambdaMART", "best_iteration": booster.best_iteration}, fh, indent=2)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(results["ranking"].keys())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for offset, metric, color in ((-0.2, "ndcg@5", "#4c72b0"), (0.0, "ndcg@10", "#dd8452"), (0.2, "map", "#55a868")):
        means = [results["ranking"][n][metric]["mean"] for n in names]
        lo = [means[i] - results["ranking"][n][metric]["ci95"][0] for i, n in enumerate(names)]
        hi = [results["ranking"][n][metric]["ci95"][1] - means[i] for i, n in enumerate(names)]
        x = np.arange(len(names)) + offset
        ax.bar(x, means, width=0.18, label=metric.upper() if metric != "map" else "MAP", color=color,
               yerr=[lo, hi], capsize=3, error_kw={"linewidth": 1})
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score (mean over 96 held-out test queries)")
    ax.set_title("Reranking quality on WANDS - mean with 95% bootstrap CI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.results_dir, "ranking_metrics.png"), dpi=140)
    plt.close(fig)

    if retrieval_out:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for name, per_k in retrieval_out.items():
            xs = [int(k.split("@")[1]) for k in per_k]
            ys = [per_k[f"recall@{k}"] for k in per_k]
            order = np.argsort(xs)
            ax.plot(np.array(xs)[order], np.array(ys)[order], marker="o", label=name)
        ax.set_xlabel("k (rank cutoff)")
        ax.set_ylabel("Recall@k (mean over test queries)")
        ax.set_title("Full-corpus retrieval recall on WANDS (42,994 products)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.results_dir, "retrieval_recall.png"), dpi=140)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    imp_items = list(results["feature_importance_gain"].items())
    ax.barh([n for n, _ in imp_items][::-1], [v for _, v in imp_items][::-1], color="#4c72b0")
    ax.set_xlabel("LightGBM split gain")
    ax.set_title("Feature importance - LambdaMART")
    fig.tight_layout()
    fig.savefig(os.path.join(args.results_dir, "feature_importance.png"), dpi=140)
    plt.close(fig)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
