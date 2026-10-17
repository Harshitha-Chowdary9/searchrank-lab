"""Run the LLM-answer evaluation on the bundled sample outputs.

Writes results/llm_eval.json comparing two mock assistants so the harness
shows a real signal: the good assistant should beat the noisy one on every
metric, and the rule-based judge should agree with the human grades.
"""
import json
import os

from .harness import RuleBasedJudge, evaluate, load_records


def main(sample_path=None, out_path="results/llm_eval.json"):
    sample_path = sample_path or os.path.join(os.path.dirname(__file__), "sample_outputs.jsonl")
    records = load_records(sample_path)
    by_model = {}
    for rec in records:
        by_model.setdefault(rec["model"], []).append(rec)
    out = {"note": "Predictions are bundled mock assistant outputs, not live LLM calls. "
                   "The harness (metrics, judge, agreement) is the deliverable.",
           "models": {}}
    for model, recs in by_model.items():
        out["models"][model] = evaluate(recs, judge=RuleBasedJudge())
        print(f"  {model:<26} token_f1 {out['models'][model]['token_f1']['mean']:.3f} "
              f"judge-agreement {out['models'][model]['judge_agreement']}", flush=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"  wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
