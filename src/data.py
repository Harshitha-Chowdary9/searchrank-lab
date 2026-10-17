"""Loading and downloading the WANDS dataset (Wayfair, ECIR 2022)."""
import csv
import os
import urllib.request

BASE_URL = "https://raw.githubusercontent.com/wayfair/WANDS/main/dataset"
FILES = ("product.csv", "query.csv", "label.csv")

GRADE = {"Exact": 2, "Partial": 1, "Irrelevant": 0}


def download_data(dest):
    os.makedirs(dest, exist_ok=True)
    for name in FILES:
        path = os.path.join(dest, name)
        if not os.path.exists(path):
            print(f"downloading {name} ...", flush=True)
            urllib.request.urlretrieve(f"{BASE_URL}/{name}", path)
    return dest


def load_products(data_dir):
    products = {}
    with open(os.path.join(data_dir, "product.csv"), newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            products[int(row["product_id"])] = {
                "name": row["product_name"] or "",
                "class": row["product_class"] or "",
                "description": row["product_description"] or "",
                "rating_count": float(row["rating_count"] or 0),
                "average_rating": float(row["average_rating"] or 0),
                "review_count": float(row["review_count"] or 0),
            }
    return products


def load_queries(data_dir):
    queries = {}
    with open(os.path.join(data_dir, "query.csv"), newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            queries[int(row["query_id"])] = row["query"]
    return queries


def load_labels(data_dir):
    """Returns {query_id: {product_id: grade}} with grade in {0,1,2}."""
    labels = {}
    with open(os.path.join(data_dir, "label.csv"), newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            qid = int(row["query_id"])
            labels.setdefault(qid, {})[int(row["product_id"])] = GRADE[row["label"]]
    return labels
