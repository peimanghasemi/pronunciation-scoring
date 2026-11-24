"""Command line: score a recording or a recognised phoneme string.

    pronscore Haus recording.wav
    pronscore Haus --phonemes "h a s"
"""
from __future__ import annotations

import argparse

from .phonemes import load_lexicon
from .score import score_attempt


def main(argv=None):
    p = argparse.ArgumentParser(prog="pronscore")
    p.add_argument("word")
    p.add_argument("audio", nargs="?")
    p.add_argument("--phonemes", help="recognised phonemes instead of audio")
    p.add_argument("--lexicon")
    a = p.parse_args(argv)

    lex = load_lexicon(a.lexicon)
    if a.word not in lex:
        p.error(f"{a.word!r} is not in the lexicon")
    conf = None
    if a.phonemes is not None:
        said = a.phonemes
    elif a.audio:
        from .recognizer import PhonemeRecognizer
        said, conf = PhonemeRecognizer()(a.audio)
    else:
        p.error("give an audio file or --phonemes")

    r = score_attempt(a.word, lex[a.word], said, confidence=conf)
    print(f"{r.word}: {r.score:.0f}/100")
    for msg in r.feedback:
        print("  -", msg)
    if r.low_confidence:
        print("  (low recognition confidence, ask for another attempt)")


if __name__ == "__main__":
    main()
