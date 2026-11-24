import numpy as np
import pytest

from pronscore import align, feature_distance, load_lexicon, score_attempt, tokenize
from pronscore.evaluate import cluster_bootstrap, quadratic_kappa, spearman
from pronscore.simulate import simulate

LEX = load_lexicon()


def test_tokenize_handles_spaces_diphthongs_and_stress():
    assert tokenize("h aʊ s") == ["h", "aʊ", "s"]
    assert tokenize("ˈhaʊs") == ["h", "aʊ", "s"]
    assert tokenize("buːx") == ["b", "uː", "x"]


def test_feature_distance():
    assert feature_distance("p", "b") == pytest.approx(1 / 3)   # voicing only
    assert feature_distance("p", "z") == 1.0
    assert feature_distance("a", "t") == 1.0
    assert feature_distance("s", "s") == 0.0


def test_perfect_attempt_scores_100():
    r = score_attempt("Haus", LEX["Haus"], "h aʊ s")
    assert r.score == 100 and r.per == 0 and r.feedback == []


def test_near_miss_costs_less_than_distant_substitution():
    near = score_attempt("Tisch", LEX["Tisch"], "t ɪ s")   # /ʃ/ -> /s/, place only
    far = score_attempt("Tisch", LEX["Tisch"], "t ɪ m")
    assert near.per == far.per
    assert near.score > far.score


def test_feedback_names_the_error():
    r = score_attempt("Brot", LEX["Brot"], "b oː t")
    assert r.feedback == ["/ʁ/ missing"]
    r = score_attempt("Ball", LEX["Ball"], "p a l")
    assert r.feedback == ["/b/ sounded like /p/"]


def test_alignment_handles_insertions_and_empty_output():
    cost, ops = align(["b", "a", "l"], ["b", "a", "a", "l"])
    assert cost == 1 and [o.kind for o in ops].count("ins") == 1
    assert score_attempt("Ball", LEX["Ball"], "").score == 0


def test_low_confidence_is_flagged():
    assert score_attempt("Haus", LEX["Haus"], "h aʊ s", confidence=0.3).low_confidence


def test_evaluation_metrics():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1)
    assert quadratic_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 5) == pytest.approx(1)
    assert quadratic_kappa([1, 1, 5, 5], [5, 5, 1, 1], 5) < 0
    g = np.repeat(np.arange(10), 5)
    x = np.arange(50.0)
    lo, hi = cluster_bootstrap(lambda x: x.mean(), g, n_boot=200, x=x)
    assert lo < x.mean() < hi


def test_simulated_scores_follow_severity():
    rows = simulate(LEX, n_speakers=8, attempts=20, seed=3)
    by = {}
    for r in rows:
        by.setdefault(r["severity"], []).append(score_attempt(r["word"], r["target"], r["recognised"]).score)
    means = [np.mean(by[s]) for s in ["none", "mild", "moderate", "severe"]]
    assert means == sorted(means, reverse=True)
