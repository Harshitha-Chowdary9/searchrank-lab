"""Feature engineering for learning-to-rank.

Every feature is computable at serving time from the query string, the
product document and corpus statistics - nothing leaks label information
and nothing needs the dataset's query_class column (a real search service
never knows the "true" category of an incoming query).
"""
import math
from .text import tokenize, bigrams

FEATURE_NAMES = [
    "bm25_score",
    "tfidf_cosine",
    "name_unigram_overlap",
    "name_bigram_overlap",
    "exact_phrase_in_name",
    "desc_unigram_coverage",
    "query_len_tokens",
    "name_len_tokens",
    "log_rating_count",
    "average_rating",
    "log_review_count",
]


def text_overlap_features(query, product):
    q_tok = tokenize(query)
    n_tok = tokenize(product["name"])
    d_tok = tokenize(product["description"])
    q_set, n_set = set(q_tok), set(n_tok)
    q_bi, n_bi = set(bigrams(q_tok)), set(bigrams(n_tok))
    name_uni = len(q_set & n_set) / len(q_set) if q_set else 0.0
    name_bi = len(q_bi & n_bi) / len(q_bi) if q_bi else 0.0
    phrase = 1.0 if query and query.lower() in (product["name"] or "").lower() else 0.0
    d_set = set(d_tok)
    desc_cov = len(q_set & d_set) / len(q_set) if q_set else 0.0
    return {
        "name_unigram_overlap": name_uni,
        "name_bigram_overlap": name_bi,
        "exact_phrase_in_name": phrase,
        "desc_unigram_coverage": desc_cov,
        "query_len_tokens": float(len(q_tok)),
        "name_len_tokens": float(len(n_tok)),
    }


def stat_features(product):
    return {
        "log_rating_count": math.log1p(product["rating_count"]),
        "average_rating": product["average_rating"],
        "log_review_count": math.log1p(product["review_count"]),
    }


def assemble_row(base, product, bm25_score, tfidf_cos):
    row = dict(base)
    row.update(stat_features(product))
    row["bm25_score"] = bm25_score
    row["tfidf_cosine"] = tfidf_cos
    return [row[name] for name in FEATURE_NAMES]
