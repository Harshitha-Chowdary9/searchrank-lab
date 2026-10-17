from src.bm25 import BM25
from src.text import tokenize


def _index():
    docs = [
        tokenize("red leather sofa with wooden legs"),
        tokenize("blue denim jeans slim fit"),
        tokenize("leather wallet brown leather"),
    ]
    return BM25(docs)


def test_relevant_doc_ranks_first():
    bm25 = _index()
    top = bm25.top_n(tokenize("leather sofa"), n=2)
    assert top[0][0] == 0


def test_unknown_term_scores_nothing():
    bm25 = _index()
    assert bm25.scores(tokenize("spaceship")) == {}


def test_idf_rewards_rare_terms():
    bm25 = _index()
    # "leather" appears in 2/3 docs, "sofa" in 1/3 -> sofa has higher idf
    assert bm25.idf["sofa"] > bm25.idf["leather"]


def test_longer_doc_normalised():
    docs = [tokenize("cat"), tokenize("cat " + "filler " * 50)]
    bm25 = BM25(docs)
    scores = bm25.scores(tokenize("cat"))
    assert scores[0] > scores[1]
