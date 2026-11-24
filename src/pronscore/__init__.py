"""Phoneme-level pronunciation scoring for speech therapy exercises."""
from .phonemes import load_lexicon, tokenize, feature_distance
from .align import align
from .score import score_attempt, Result

__all__ = ["load_lexicon", "tokenize", "feature_distance", "align", "score_attempt", "Result"]
__version__ = "0.2.0"
