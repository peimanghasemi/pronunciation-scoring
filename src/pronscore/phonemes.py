"""Phoneme inventory, articulatory features and the word lexicon."""
from __future__ import annotations

import csv
from importlib import resources

# consonants: (place, manner, voiced)
CONSONANTS = {
    "p": ("bilabial", "plosive", 0), "b": ("bilabial", "plosive", 1),
    "t": ("alveolar", "plosive", 0), "d": ("alveolar", "plosive", 1),
    "k": ("velar", "plosive", 0), "g": ("velar", "plosive", 1),
    "m": ("bilabial", "nasal", 1), "n": ("alveolar", "nasal", 1), "ŋ": ("velar", "nasal", 1),
    "f": ("labiodental", "fricative", 0), "v": ("labiodental", "fricative", 1),
    "s": ("alveolar", "fricative", 0), "z": ("alveolar", "fricative", 1),
    "ʃ": ("postalveolar", "fricative", 0), "ʒ": ("postalveolar", "fricative", 1),
    "ç": ("palatal", "fricative", 0), "x": ("velar", "fricative", 0),
    "ʁ": ("uvular", "fricative", 1), "h": ("glottal", "fricative", 0),
    "l": ("alveolar", "lateral", 1), "j": ("palatal", "approximant", 1),
}
# vowels: (height, backness, rounded, long)
VOWELS = {
    "iː": ("close", "front", 0, 1), "ɪ": ("near-close", "front", 0, 0),
    "yː": ("close", "front", 1, 1), "ʏ": ("near-close", "front", 1, 0),
    "eː": ("close-mid", "front", 0, 1), "ɛ": ("open-mid", "front", 0, 0),
    "øː": ("close-mid", "front", 1, 1), "œ": ("open-mid", "front", 1, 0),
    "uː": ("close", "back", 1, 1), "ʊ": ("near-close", "back", 1, 0),
    "oː": ("close-mid", "back", 1, 1), "o": ("close-mid", "back", 1, 0),
    "ɔ": ("open-mid", "back", 1, 0),
    "aː": ("open", "central", 0, 1), "a": ("open", "central", 0, 0),
    "ə": ("mid", "central", 0, 0), "ɐ": ("near-open", "central", 0, 0),
    "aʊ": ("open", "diphthong", 1, 1), "aɪ": ("open", "diphthong", 0, 1),
    "ɔʏ": ("open-mid", "diphthong", 1, 1),
}
INVENTORY = {**CONSONANTS, **VOWELS}
_BY_LENGTH = sorted(INVENTORY, key=len, reverse=True)
_IGNORE = set("ˈˌ.|")


def tokenize(s: str) -> list[str]:
    """Split a phoneme string into phonemes.

    Accepts space separated input ("h aʊ s") as well as unseparated IPA
    ("haʊs"), the way phoneme recognisers often return it. Stress marks are
    dropped; unknown symbols are kept as their own token.
    """
    out = []
    for chunk in s.split():
        i = 0
        while i < len(chunk):
            if chunk[i] in _IGNORE:
                i += 1
                continue
            for p in _BY_LENGTH:
                if chunk.startswith(p, i):
                    out.append(p)
                    i += len(p)
                    break
            else:
                out.append(chunk[i])
                i += 1
    return out


def feature_distance(a: str, b: str) -> float:
    """Share of articulatory features in which two phonemes differ (0 to 1).

    Consonant against vowel, or an unknown symbol, counts as fully different.
    """
    if a == b:
        return 0.0
    if a in CONSONANTS and b in CONSONANTS:
        fa, fb = CONSONANTS[a], CONSONANTS[b]
    elif a in VOWELS and b in VOWELS:
        fa, fb = VOWELS[a], VOWELS[b]
    else:
        return 1.0
    return sum(x != y for x, y in zip(fa, fb)) / len(fa)


def load_lexicon(path: str | None = None) -> dict[str, list[str]]:
    """Target words and their phonemes (German therapy word list by default)."""
    if path is None:
        text = resources.files("pronscore.data").joinpath("lexicon_de.csv").read_text(encoding="utf-8")
    else:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    rows = csv.DictReader(text.splitlines())
    return {r["word"]: tokenize(r["phonemes"]) for r in rows}
