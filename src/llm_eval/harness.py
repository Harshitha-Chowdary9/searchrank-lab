"""Evaluation harness for LLM-generated answers.

Three complementary evaluation styles, all runnable fully offline:

1. Reference-based string metrics (exact match, token F1, ROUGE-L) - cheap
   and deterministic, but blind to paraphrases.
2. A judge interface that scores an answer 0/1/2 against a rubric. The
   default RuleBasedJudge is deterministic and offline; an
   OpenAICompatibleJudge can call any chat-completions endpoint when an API
   key is configured (never required for tests).
3. Agreement analysis (percent agreement + Cohen's kappa) between the judge
   and the human reference grades - the standard sanity check before
   trusting an automated judge.
"""
import json
import os
import urllib.request
from collections import Counter

from ..metrics import bootstrap_ci
from ..text import tokenize


def exact_match(prediction, reference):
    return float((prediction or "").strip().lower() == (reference or "").strip().lower())


def token_f1(prediction, reference):
    p, r = tokenize(prediction), tokenize(reference)
    if not p or not r:
        return 0.0
    common = Counter(p) & Counter(r)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(p)
    recall = n_common / len(r)
    return 2 * precision * recall / (precision + recall)


def _lcs_len(a, b):
    prev = [0] * (len(b) + 1)
    for tok_a in a:
        cur = [0]
        for j, tok_b in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if tok_a == tok_b else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l(prediction, reference):
    p, r = tokenize(prediction), tokenize(reference)
    if not p or not r:
        return 0.0
    lcs = _lcs_len(p, r)
    precision = lcs / len(p)
    recall = lcs / len(r)
    return 0.0 if lcs == 0 else 2 * precision * recall / (precision + recall)


class RuleBasedJudge:
    """Deterministic offline judge implementing a fixed rubric:

    grade 2 (good):       key content words of the reference are covered and
                          no reference number is contradicted
    grade 1 (partial):    topical overlap but missing/contradicting detail
    grade 0 (wrong):      little overlap with the reference
    """

    def score(self, question, prediction, reference):
        ref_tokens = set(tokenize(reference))
        pred_tokens = set(tokenize(prediction))
        if not ref_tokens or not pred_tokens:
            return 0
        coverage = len(ref_tokens & pred_tokens) / len(ref_tokens)
        ref_nums = {t for t in ref_tokens if t.isdigit()}
        contradicted = bool(ref_nums) and not ref_nums.issubset(pred_tokens)
        if coverage >= 0.6 and not contradicted:
            return 2
        if coverage >= 0.3:
            return 1
        return 0


class OpenAICompatibleJudge:
    """Optional judge calling any OpenAI-compatible /chat/completions API.

    Only used when LLM_API_KEY is set; the repo's tests and reported results
    never depend on it.
    """

    PROMPT = ("Grade the ANSWER against the REFERENCE for the QUESTION. "
              "Reply with one digit: 2 = fully correct, 1 = partially correct, "
              "0 = wrong.\nQUESTION: {q}\nREFERENCE: {r}\nANSWER: {a}")

    def __init__(self, base_url=None, model=None):
        self.base_url = base_url or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self.api_key = os.environ.get("LLM_API_KEY", "")

    def score(self, question, prediction, reference):
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": self.PROMPT.format(q=question, r=reference, a=prediction)}],
            "temperature": 0,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = json.load(resp)["choices"][0]["message"]["content"].strip()
        return int(text[0]) if text and text[0] in "012" else 0


def cohens_kappa(grades_a, grades_b):
    """Inter-rater agreement beyond chance for two aligned grade lists."""
    n = len(grades_a)
    if n == 0:
        return 0.0
    labels = sorted(set(grades_a) | set(grades_b))
    observed = sum(a == b for a, b in zip(grades_a, grades_b)) / n
    count_a = Counter(grades_a)
    count_b = Counter(grades_b)
    expected = sum((count_a[l] / n) * (count_b[l] / n) for l in labels)
    return 1.0 if expected == 1.0 else (observed - expected) / (1 - expected)


def evaluate(records, judge=None, seed=13):
    """records: [{"question","reference","prediction","human_grade"}...]
    Returns aggregate metrics with bootstrap CIs plus judge agreement."""
    judge = judge or RuleBasedJudge()
    rows = []
    for rec in records:
        rows.append({
            "exact_match": exact_match(rec["prediction"], rec["reference"]),
            "token_f1": token_f1(rec["prediction"], rec["reference"]),
            "rouge_l": rouge_l(rec["prediction"], rec["reference"]),
            "judge_grade": judge.score(rec["question"], rec["prediction"], rec["reference"]),
            "human_grade": rec.get("human_grade"),
        })
    out = {}
    for metric in ("exact_match", "token_f1", "rouge_l"):
        vals = [r[metric] for r in rows]
        lo, hi = bootstrap_ci(vals, seed=seed)
        out[metric] = {"mean": round(sum(vals) / len(vals), 4), "ci95": [round(lo, 4), round(hi, 4)]}
    human = [r["human_grade"] for r in rows]
    if all(g is not None for g in human):
        grades = [r["judge_grade"] for r in rows]
        out["judge_agreement"] = {
            "percent_agreement": round(sum(a == b for a, b in zip(grades, human)) / len(rows), 4),
            "cohens_kappa": round(cohens_kappa(grades, human), 4),
        }
        out["mean_judge_grade"] = round(sum(grades) / len(grades), 4)
    return out


def load_records(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
