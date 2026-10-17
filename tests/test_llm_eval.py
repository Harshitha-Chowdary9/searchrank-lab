import math

from src.llm_eval.harness import (
    RuleBasedJudge,
    cohens_kappa,
    evaluate,
    rouge_l,
    token_f1,
)


def test_token_f1_identical():
    assert token_f1("the sofa is red", "the sofa is red") == 1.0


def test_token_f1_disjoint():
    assert token_f1("sofa", "table") == 0.0


def test_rouge_l_identical():
    assert rouge_l("free returns within 30 days", "free returns within 30 days") == 1.0


def test_judge_rewards_coverage():
    judge = RuleBasedJudge()
    grade = judge.score("q", "Delivery takes 5 days and returns are free",
                        "Delivery takes 5 days and returns are free")
    assert grade == 2


def test_judge_flags_contradicted_numbers():
    judge = RuleBasedJudge()
    grade = judge.score("q", "Delivery takes 9 days and returns are free",
                        "Delivery takes 5 days and returns are free")
    assert grade < 2


def test_kappa_perfect_agreement():
    assert math.isclose(cohens_kappa([0, 1, 2, 2], [0, 1, 2, 2]), 1.0)


def test_kappa_constant_rater_is_zero():
    assert math.isclose(cohens_kappa([1, 1, 1, 1], [0, 1, 2, 1]), 0.0, abs_tol=1e-9)


def test_evaluate_ranks_better_model_higher():
    records = [
        {"question": "q1", "reference": "returns are free within 30 days",
         "prediction": "returns are free within 30 days", "human_grade": 2},
        {"question": "q2", "reference": "the sofa costs 499 euros",
         "prediction": "I do not know", "human_grade": 0},
    ]
    out = evaluate(records)
    assert out["token_f1"]["mean"] > 0.4
    assert out["judge_agreement"]["percent_agreement"] >= 0.5
