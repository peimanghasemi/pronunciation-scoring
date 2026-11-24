"""Pronunciation score and phoneme-level feedback for one attempt."""
from __future__ import annotations

from dataclasses import dataclass, field

from .align import Op, align
from .phonemes import tokenize


@dataclass
class Result:
    word: str
    score: float                 # 0 to 100
    per: float                   # plain phoneme error rate
    ops: list[Op] = field(repr=False)
    feedback: list[str]
    low_confidence: bool = False


def _message(op: Op) -> str:
    if op.kind == "sub":
        return f"/{op.target}/ sounded like /{op.said}/"
    if op.kind == "del":
        return f"/{op.target}/ missing"
    return f"extra /{op.said}/"


def score_attempt(word: str, target: list[str] | str, said: list[str] | str,
                  weighted: bool = True, confidence: float | None = None,
                  min_confidence: float = 0.5) -> Result:
    """Score a recognised attempt against the target word.

    The score is 100 * (1 - alignment cost / number of target phonemes),
    floored at 0. `per` is the classic unweighted phoneme error rate. If the
    recogniser's confidence is below `min_confidence`, the result is flagged
    so the app can ask for another attempt instead of giving wrong feedback.
    """
    t = tokenize(target) if isinstance(target, str) else list(target)
    s = tokenize(said) if isinstance(said, str) else list(said)
    if not t:
        raise ValueError(f"no target phonemes for {word!r}")
    cost, ops = align(t, s, weighted=weighted)
    plain, _ = align(t, s, weighted=False)
    return Result(
        word=word,
        score=round(100 * max(0.0, 1 - cost / len(t)), 1),
        per=plain / len(t),
        ops=ops,
        feedback=[_message(o) for o in ops if o.kind != "ok"],
        low_confidence=confidence is not None and confidence < min_confidence,
    )
