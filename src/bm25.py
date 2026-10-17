"""BM25 (Okapi) written from scratch with an inverted index.

Same spirit as the TF-IDF-from-scratch in the sibling banking-doc-rag
project: no retrieval library does the work for us, so every piece of the
scoring can be explained in an interview.
"""
import heapq
import math
from collections import Counter, defaultdict


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        """docs: list of token lists (one per document)."""
        self.k1 = k1
        self.b = b
        self.n_docs = len(docs)
        self.doc_len = [len(d) for d in docs]
        self.avgdl = sum(self.doc_len) / max(1, self.n_docs)
        df = defaultdict(int)
        self.postings = defaultdict(list)  # term -> [(doc_id, tf), ...]
        for doc_id, tokens in enumerate(docs):
            counts = Counter(tokens)
            for term, tf in counts.items():
                df[term] += 1
                self.postings[term].append((doc_id, tf))
        # Robertson/Spärck-Jones idf, kept non-negative (+1 inside the log).
        self.idf = {
            t: math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))
            for t, n in df.items()
        }

    def scores(self, query_tokens):
        """Return {doc_id: bm25_score} for docs sharing a term with the query."""
        out = defaultdict(float)
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_id, tf in self.postings[term]:
                norm = 1 - self.b + self.b * self.doc_len[doc_id] / self.avgdl
                out[doc_id] += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * norm)
        return dict(out)

    def top_n(self, query_tokens, n=100):
        """Return [(doc_id, score)] sorted by score, best first."""
        return heapq.nlargest(n, self.scores(query_tokens).items(), key=lambda kv: kv[1])
