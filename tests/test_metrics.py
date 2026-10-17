import math

from src.metrics import (
    average_precision,
    bootstrap_ci,
    dcg_at_k,
    mean_metric,
    ndcg_at_k,
    paired_bootstrap_diff,
    per_query_metrics,
    reciprocal_rank,
)


def test_dcg_perfect_order():
    # grades [2, 1]: gain 3 / log2(2) + 1 / log2(3)
    assert math.isclose(dcg_at_k([2, 1], 2), 3.0 + 1.0 / math.log2(3))


def test_ndcg_perfect_is_one():
    assert ndcg_at_k([2, 1, 0], 3) == 1.0


def test_ndcg_reversed_below_one():
    assert 0.0 < ndcg_at_k([0, 1, 2], 3) < 1.0


def test_ndcg_all_irrelevant_is_zero():
    assert ndcg_at_k([0, 0, 0], 10) == 0.0


def test_average_precision():
    # relevant at ranks 1 and 3 of 3 -> (1/1 + 2/3) / 2
    assert math.isclose(average_precision([1, 0, 2]), (1.0 + 2.0 / 3.0) / 2.0)


def test_average_precision_none_when_no_relevant():
    assert average_precision([0, 0]) is None


def test_reciprocal_rank():
    assert reciprocal_rank([0, 0, 1]) == 1.0 / 3.0
    assert reciprocal_rank([0, 0]) is None


def test_per_query_metrics_unjudged_counts_as_zero():
    ranked = {0: [10, 11, 12]}
    labels = {0: {10: 2}}  # 11, 12 unjudged -> grade 0
    pq = per_query_metrics(ranked, labels)
    assert pq[0]["ndcg@5"] == 1.0
    assert pq[0]["mrr"] == 1.0


def test_bootstrap_ci_contains_mean():
    vals = [0.5] * 50 + [0.7] * 50
    lo, hi = bootstrap_ci(vals, n_boot=200)
    assert lo <= 0.6 <= hi


def test_paired_bootstrap_detects_clear_improvement():
    a = [0.9] * 30
    b = [0.1] * 30
    lo, hi = paired_bootstrap_diff(a, b, n_boot=200)
    assert lo > 0.0
