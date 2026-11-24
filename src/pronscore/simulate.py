"""Simulated therapy attempts for testing the scoring and the evaluation.

Real patient recordings are not public, so the evaluation pipeline is
exercised on simulated recogniser output. Each speaker has a severity that
sets how often phonemes are substituted, dropped or added. Substitutions
favour phonetically close phonemes, as is typical for dysarthric speech.
The recogniser itself also makes mistakes, more often for severe speech,
and its confidence drops when it does.

Clinician ratings (1 = unintelligible, 5 = correct) are generated from the
true production errors, with near misses hurting intelligibility less than
distant ones, plus rater noise. The scorer never sees the true errors, only
the noisy recogniser output.
"""
from __future__ import annotations

import numpy as np

from .phonemes import INVENTORY, feature_distance

SEVERITY = {"none": 0.02, "mild": 0.10, "moderate": 0.22, "severe": 0.38}
ASR_ERROR = {"none": 0.03, "mild": 0.05, "moderate": 0.08, "severe": 0.12}
_PHONES = list(INVENTORY)


def _near(p: str, rng) -> str:
    """Pick a substitute, closer phonemes being more likely."""
    cands = [q for q in _PHONES if q != p]
    d = np.array([feature_distance(p, q) for q in cands])
    w = np.exp(-4 * d)
    return cands[rng.choice(len(cands), p=w / w.sum())]


def corrupt(seq: list[str], rate: float, rng) -> tuple[list[str], float, int]:
    """Apply substitutions (70%), deletions (20%) and insertions (10%).

    Returns the new sequence, an intelligibility loss in which a
    substitution counts by its phonetic distance and indels count fully,
    and the number of changes.
    """
    out, loss, n = [], 0.0, 0
    for p in seq:
        if rng.random() < rate:
            n += 1
            r = rng.random()
            if r < 0.7:
                q = _near(p, rng)
                out.append(q)
                loss += 0.3 + 0.7 * feature_distance(p, q)
            elif r < 0.9:
                loss += 1.0
            else:
                out += [p, rng.choice(_PHONES)]
                loss += 0.5
        else:
            out.append(p)
    return out, loss, n


def simulate(lexicon: dict[str, list[str]], n_speakers: int = 40,
             attempts: int = 30, seed: int = 1) -> list[dict]:
    rng = np.random.default_rng(seed)
    levels = list(SEVERITY)
    words = list(lexicon)
    rows = []
    for s in range(n_speakers):
        sev = levels[s % len(levels)]
        rate = SEVERITY[sev] * rng.uniform(0.7, 1.3)
        for _ in range(attempts):
            w = words[rng.integers(len(words))]
            target = lexicon[w]
            produced, loss, _ = corrupt(target, rate, rng)
            heard, _, asr_errors = corrupt(produced, ASR_ERROR[sev], rng)
            latent = 1 - loss / len(target) + rng.normal(0, 0.08)
            rating = int(np.digitize(latent, [0.35, 0.6, 0.78, 0.92]) + 1)
            # the recogniser tends to be less sure when it gets things wrong
            confidence = float(np.clip(0.88 - 0.6 * asr_errors / len(target)
                                       - ASR_ERROR[sev] + rng.normal(0, 0.07), 0, 1))
            rows.append(dict(speaker=f"S{s:02d}", severity=sev, word=w, target=target,
                             recognised=heard, rating=rating, confidence=confidence))
    return rows
