import os
os.environ.setdefault("SEARCHRANK_SKIP_AUTOSTART", "1")
"""API test on a tiny synthetic corpus with a real (tiny) LightGBM model."""
import csv
import json
import os

import numpy as np
import pytest

lgb = pytest.importorskip("lightgbm")
from fastapi.testclient import TestClient

from src.api import create_app
from src.features import FEATURE_NAMES


def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(header)
        w.writerows(rows)


@pytest.fixture()
def tiny_app(tmp_path):
    data_dir = tmp_path / "data"
    art_dir = tmp_path / "artifacts"
    data_dir.mkdir()
    art_dir.mkdir()
    _write_csv(
        data_dir / "product.csv",
        ["product_id", "product_name", "product_class", "category hierarchy",
         "product_description", "product_features", "rating_count", "average_rating", "review_count"],
        [
            [1, "red leather sofa", "Sofas", "", "comfortable red leather sofa", "", 5, 4.0, 4],
            [2, "blue denim jeans", "Jeans", "", "slim fit blue jeans", "", 2, 3.5, 2],
            [3, "wooden dining table", "Tables", "", "solid oak dining table", "", 9, 4.8, 7],
        ],
    )
    # train a tiny LambdaMART on synthetic feature rows
    rng = np.random.RandomState(0)
    X = rng.rand(30, len(FEATURE_NAMES)).astype(np.float32)
    y = np.array([2] * 10 + [1] * 10 + [0] * 10, dtype=np.int32)
    ds = lgb.Dataset(X, label=y, group=[10, 10, 10], feature_name=FEATURE_NAMES)
    booster = lgb.train({"objective": "lambdarank", "verbosity": -1, "num_leaves": 3}, ds, num_boost_round=5)
    booster.save_model(str(art_dir / "lgbm_ranker.txt"))
    with open(art_dir / "feature_config.json", "w") as fh:
        json.dump({"feature_names": FEATURE_NAMES, "bm25_k1": 1.5, "bm25_b": 0.75}, fh)
    return create_app(str(data_dir), str(art_dir))


def test_health(tiny_app):
    with TestClient(tiny_app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["products"] == 3


def test_search_returns_ranked_results(tiny_app):
    with TestClient(tiny_app) as client:
        resp = client.post("/search", json={"query": "leather sofa", "k": 3})
        assert resp.status_code == 200
        body = resp.json()
        assert body["results"], "expected at least one result"
        names = [r["name"] for r in body["results"]]
        assert "red leather sofa" in names
        # the matching product should be the top candidate after reranking
        assert names[0] == "red leather sofa"


def test_search_validates_empty_query(tiny_app):
    with TestClient(tiny_app) as client:
        assert client.post("/search", json={"query": ""}).status_code == 422


def test_metrics_counts_searches(tiny_app):
    with TestClient(tiny_app) as client:
        client.post("/search", json={"query": "table"})
        metrics = client.get("/metrics").json()
        assert metrics["searches"] == 1
