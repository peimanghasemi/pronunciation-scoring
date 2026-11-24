"""Alignment of a recognised phoneme sequence against the target."""
from __future__ import annotations

from dataclasses import dataclass

from .phonemes import feature_distance


@dataclass
class Op:
    kind: str            # "ok", "sub", "del", "ins"
    target: str | None
    said: str | None
    cost: float


def sub_cost(a: str, b: str, weighted: bool = True) -> float:
    """Substitution cost. With weighting, a near miss (e.g. /b/ for /p/,
    one feature off) costs less than an unrelated phoneme."""
    if a == b:
        return 0.0
    if not weighted:
        return 1.0
    return 0.4 + 0.6 * feature_distance(a, b)


def align(target: list[str], said: list[str], weighted: bool = True,
          indel: float = 1.0) -> tuple[float, list[Op]]:
    """Minimum-cost alignment (Levenshtein with phonetic substitution costs).

    Returns the total cost and the list of operations in target order.
    """
    n, m = len(target), len(said)
    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = i * indel
    for j in range(1, m + 1):
        D[0][j] = j * indel
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j - 1] + sub_cost(target[i - 1], said[j - 1], weighted),
                          D[i - 1][j] + indel,
                          D[i][j - 1] + indel)

    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            c = sub_cost(target[i - 1], said[j - 1], weighted)
            if abs(D[i][j] - (D[i - 1][j - 1] + c)) < 1e-12:
                ops.append(Op("ok" if c == 0 else "sub", target[i - 1], said[j - 1], c))
                i, j = i - 1, j - 1
                continue
        if i > 0 and abs(D[i][j] - (D[i - 1][j] + indel)) < 1e-12:
            ops.append(Op("del", target[i - 1], None, indel))
            i -= 1
        else:
            ops.append(Op("ins", None, said[j - 1], indel))
            j -= 1
    return D[n][m], ops[::-1]
