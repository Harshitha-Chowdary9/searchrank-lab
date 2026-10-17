"""Ranking metrics + bootstrap confidence intervals, all from scratch."""
import math
import random


def dcg_at_k(grades, k):
    """DCG with exponential gain (2^rel - 1) and log2 position discount."""
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades[:k]))


def ndcg_at_k(ranked_grades, k):
    """ranked_grades: relevance grades in ranked order."""
    idcg = dcg_at_k(sorted(ranked_grades, reverse=True), k)
    return dcg_at_k(ranked_grades, k) / idcg if idcg > 0 else 0.0


def average_precision(ranked_grades):
    """Binary AP; a document counts as relevant when grade >= 1.

    Returns None for queries with no relevant documents (excluded from MAP).
    """
    hits = 0
    total = 0.0
    n_rel = sum(1 for g in ranked_grades if g >= 1)
    if n_rel == 0:
        return None
    for i, g in enumerate(ranked_grades):
        if g >= 1:
            hits += 1
            total += hits / (i + 1)
    return total / n_rel


def reciprocal_rank(ranked_grades):
    for i, g in enumerate(ranked_grades):
        if g >= 1:
            return 1.0 / (i + 1)
    return None


def per_query_metrics(ranked_lists, labels, ks=(5, 10)):
    """ranked_lists: {qid: [product_id, ...] ranked best-first}
    labels: {qid: {product_id: grade}}
    Returns {qid: {"ndcg@k": ..., "map": ..., "mrr": ...}}.
    Unjudged retrieved documents count as grade 0 (standard, conservative).
    """
    out = {}
    for qid, ranked in ranked_lists.items():
        lab = labels.get(qid, {})
        grades = [lab.get(pid, 0) for pid in ranked]
        row = {f"ndcg@{k}": ndcg_at_k(grades, k) for k in ks}
        row["map"] = average_precision(grades)
        row["mrr"] = reciprocal_rank(grades)
        out[qid] = row
    return out


def mean_metric(per_query, name):
    vals = [m[name] for m in per_query.values() if m.get(name) is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _percentile_ci(draws, alpha):
    draws.sort()
    n = len(draws)
    lo = draws[int((alpha / 2) * n)]
    hi = draws[min(n - 1, int((1 - alpha / 2) * n))]
    return (lo, hi)


def bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=13):
    """Percentile bootstrap CI of the mean over query-level values."""
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    means = []
    for _ in range(n_boot):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    return _percentile_ci(means, alpha)


def paired_bootstrap_diff(values_a, values_b, n_boot=1000, alpha=0.05, seed=29):
    """CI of mean(a - b) over paired query-level values (aligned by index).

    If the CI excludes 0, the difference is significant at level alpha.
    """
    rng = random.Random(seed)
    n = len(values_a)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(values_a[i] - values_b[i] for i in idx) / n)
    return _percentile_ci(diffs, alpha)
