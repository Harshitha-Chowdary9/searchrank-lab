# Study guide - interview questions this project prepares you for

Read this alongside the code. Every answer points at the file where you can
see the idea working.

## Retrieval

**Q: How does BM25 improve on plain TF-IDF?**
BM25 saturates term frequency (a word occurring 10 times is not 10x as
important as once, thanks to the k1 curve) and normalises for document
length (parameter b), so long descriptions do not win just by containing
more words. See `src/bm25.py`; the length-normalisation effect is covered
by `test_longer_doc_normalised`.

**Q: Sparse (BM25) vs dense (embeddings) retrieval - when does each win?**
BM25 is exact-match: great for part numbers, rare words, brand names; it
fails on synonyms ("couch" vs "sofa"). Dense embeddings capture semantic
similarity but can blur exact constraints. In this project we measured
both on the same test queries, plus a min-max-normalised hybrid that fuses
the two score scales - check `results/metrics.json` under `retrieval` and
`results/retrieval_recall.png`.

**Q: Why is Recall@k the right retrieval metric here?**
Retrieval is a candidate-generation step: the reranker can only fix what
it sees. Recall@k measures how much of the relevant material survives the
cutoff, which is exactly the property the reranker depends on.

## Learning to rank

**Q: Pointwise vs pairwise vs listwise - what did you implement and why?**
All three flavours are represented: logistic regression scores each
document independently (pointwise), the PyTorch RankNet minimises a
pairwise logistic loss over misordered pairs (`src/models/torch_ranker.py`),
and LightGBM's LambdaMART optimises a listwise objective (lambdarank) that
weights swaps by their NDCG impact. In interviews: pointwise ignores
relative order; pairwise cares about order but weights all pairs equally;
LambdaMART focuses gradient on pairs that change the top of the ranking.

**Q: Why did you split by query instead of by row?**
Rows from the same query share the query text. A random row split leaks the
query into train and test and inflates every metric. Query-level splitting
(`split_queries` in `src/train.py`) tests generalisation to *unseen
queries*, which is the production setting.

**Q: Which features mattered and why?**
`results/feature_importance.png` shows LightGBM gain per feature. Expect
BM25 and phrase/overlap features to dominate - they capture exact lexical
relevance - with smaller contributions from description coverage and the
product-quality signals (rating counts). The takeaway: simple lexical
features are hard to beat in e-commerce search.

**Q: Why did you not use the dataset's `query_class` column as a feature?**
It is the *true* category of the query - label information a real search
service never has at query time. Using it would leak. This is the kind of
leakage interviewers probe for.

## Evaluation

**Q: NDCG vs MAP vs MRR?**
NDCG handles graded relevance (Exact > Partial) and discounts by rank
position - the main metric here. MAP is binary relevance, rewards ranking
*all* relevant items highly. MRR only cares about the first relevant hit
(navigational queries). We report all three; they tell different stories.

**Q: Why bootstrap confidence intervals?**
Per-query metric values are not i.i.d. and their distribution is skewed,
so a normal CI is wrong. Resampling queries with replacement
(`bootstrap_ci` in `src/metrics.py`) needs no distributional assumption.
The paired bootstrap (`paired_bootstrap_diff`) compares two rankers on the
*same* queries, which removes between-query variance - that is why it can
call a 2-point NDCG difference significant.

**Q: How would you evaluate an LLM answer when references exist and when they do not?**
With references: token F1 / ROUGE-L (cheap, deterministic, blind to
paraphrase) - see `src/llm_eval/harness.py`. Without: an LLM-as-judge with
a fixed rubric, but you must validate the judge against human grades first
- that is what the percent-agreement and Cohen's kappa calculations do.
Never trust a judge you have not measured.

## Engineering

**Q: Walk through what happens when /search is called.**
Tokenise query -> BM25 top-N candidates over the inverted index ->
per-candidate feature extraction -> LambdaMART scores -> re-sort -> top-k
JSON. Latency is measured and exposed on /metrics (`src/api.py`).

**Q: Why rebuild the BM25/TF-IDF index at startup instead of pickling it?**
The corpus is a public CSV; rebuilding takes seconds and removes a whole
class of stale-artifact bugs (index built from a different data version
than the model). The trained model, by contrast, is an artifact - you do
not retrain at startup.

**Q: What would you add before calling this production?**
Real query logs with click-based labels, ANN index (FAISS/HNSW) for dense
retrieval at scale, feature store, online A/B evaluation, model monitoring,
and caching. The README lists these honestly as not-done.
