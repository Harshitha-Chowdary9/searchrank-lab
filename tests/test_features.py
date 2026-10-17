from src.features import FEATURE_NAMES, assemble_row, text_overlap_features


PRODUCT = {
    "name": "Modern Leather Sofa",
    "description": "A comfortable three seat sofa in genuine leather",
    "rating_count": 10.0,
    "average_rating": 4.5,
    "review_count": 8.0,
}


def test_full_overlap():
    feats = text_overlap_features("leather sofa", PRODUCT)
    assert feats["name_unigram_overlap"] == 1.0
    assert feats["name_bigram_overlap"] == 1.0
    assert feats["exact_phrase_in_name"] == 1.0


def test_no_overlap():
    feats = text_overlap_features("dining table", PRODUCT)
    assert feats["name_unigram_overlap"] == 0.0
    assert feats["exact_phrase_in_name"] == 0.0


def test_assemble_row_order():
    row = assemble_row(text_overlap_features("sofa", PRODUCT), PRODUCT, 3.2, 0.4)
    assert len(row) == len(FEATURE_NAMES)
    assert row[FEATURE_NAMES.index("bm25_score")] == 3.2
    assert row[FEATURE_NAMES.index("tfidf_cosine")] == 0.4
