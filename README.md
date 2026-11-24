# pronunciation-scoring

Phoneme-level pronunciation scoring for speech therapy exercises. A patient says a target word, a phoneme recogniser (wav2vec2 CTC) turns the recording into phonemes, and the attempt is scored against the target with feedback a therapist or an app can use, such as "/ʃ/ sounded like /s/" or "/ʁ/ missing".

The repository is deliberately small. It covers the parts around the acoustic model that decide whether the scores can be trusted in a clinical setting: how errors are weighted, how low-quality recognitions are caught, and how agreement with clinician ratings is measured.

## How scoring works

1. **Recognition** (`recognizer.py`). Uses `facebook/wav2vec2-lv-60-espeak-cv-ft` by default, a multilingual CTC model that outputs IPA phonemes. It also returns a confidence value, the mean top probability over non-blank frames.
2. **Alignment** (`align.py`). Levenshtein alignment of the recognised phonemes against the target. Substitutions are weighted by articulatory features (place, manner and voicing for consonants; height, backness, rounding and length for vowels). /b/ for /p/ is a near miss, while /m/ for /ʃ/ is not.
3. **Score and feedback** (`score.py`). The score is 0 to 100 based on the weighted alignment cost, with the plain phoneme error rate reported alongside. Each non-matching step becomes a feedback message. Attempts with low recognition confidence are flagged, so the app can ask for a repeat instead of giving wrong feedback.

The target words come from a small German therapy word list (`data/lexicon_de.csv`, 25 words: one and two syllables, clusters, affricates). Any CSV with `word,phonemes` works.

```bash
pip install -e .                       # scoring and evaluation, only needs numpy
pip install -e ".[audio]"              # plus torch, transformers, librosa for audio

pronscore Tisch --phonemes "t ɪ s"
# Tisch: 80/100
#   - /ʃ/ sounded like /s/

pronscore Haus recording.wav           # with the audio extras
```

```python
from pronscore import load_lexicon, score_attempt
lex = load_lexicon()
r = score_attempt("Brot", lex["Brot"], "b oː t")
r.score, r.feedback        # (75.0, ['/ʁ/ missing'])
```

## Evaluation

Patient recordings with clinician ratings are not public, so the evaluation runs on simulated attempts (`simulate.py`).
- 40 speakers at four severity levels.
- Production errors favour phonetically close substitutions, as is typical for dysarthric speech.
- A recogniser adds its own errors, and its confidence drops when it makes them.
- Clinician ratings from 1 to 5 are derived from the true production errors plus rater noise. The scorer only sees the noisy recogniser output.

Confidence intervals come from a bootstrap over speakers rather than over attempts, since attempts from the same person are not independent.

| Scoring | Spearman with ratings | Weighted kappa |
|---|---|---|
| Plain phoneme error rate | 0.71 (0.65 to 0.75) | 0.66 (0.61 to 0.70) |
| Phonetically weighted | 0.71 (0.66 to 0.75) | 0.72 (0.67 to 0.76) |

- **Ranking.** The two scores rank attempts almost identically (difference 0.004, interval -0.005 to 0.015).
- **Calibration.** The weighted score agrees clearly better with the rating scale (kappa 0.72 against 0.66), because near misses are no longer punished as hard as missing sounds. This matters when the score is shown to patients as feedback.
- **Severity.** Exact agreement falls from 0.77 for unimpaired speakers to 0.46 for severe cases, while agreement within one point stays at 0.90 or above in every group.
- **Confidence flag.** The 3.7% of attempts it catches have an exact agreement of only 0.20, against 0.60 for the rest. Asking for a repeat on those attempts is better than scoring them.

Full tables are in `results/evaluation.md`. Because the ratings are simulated, the numbers show what the evaluation measures and how. They are not a claim about real patients. With real data, the same script reports the same statistics.

```bash
python scripts/run_evaluation.py
pytest
```

## Next steps with real data

- Fine-tune the recogniser on impaired speech (for example TORGO or UASpeech, or in-house data) with LoRA adapters on the attention layers, and compare per-severity phoneme error rates before and after.
- Learn the substitution weights from clinician ratings instead of setting them from features.
- Report agreement per diagnosis group and per word type, not only overall, so weaknesses on clusters or long words are visible.

## License

MIT
