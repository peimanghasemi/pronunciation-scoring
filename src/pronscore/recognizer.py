"""Phoneme recognition from audio with a wav2vec2 CTC model (optional).

Needs the extra dependencies: pip install "pronscore[audio]"
The default model is facebook/wav2vec2-lv-60-espeak-cv-ft, a multilingual
phoneme recogniser that outputs IPA, so its output can be scored directly
against the lexicon. Any CTC model with a phoneme vocabulary works.
"""
from __future__ import annotations

DEFAULT_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"


class PhonemeRecognizer:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        import torch
        from transformers import AutoModelForCTC, AutoProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForCTC.from_pretrained(model_name).to(self.device).eval()

    def load(self, path: str):
        import librosa
        audio, _ = librosa.load(path, sr=16000, mono=True)
        return audio

    def __call__(self, audio) -> tuple[str, float]:
        """Return (phoneme string, confidence).

        Confidence is the mean top probability over frames that are not
        CTC blanks, a cheap signal for recordings the model cannot handle.
        """
        if isinstance(audio, str):
            audio = self.load(audio)
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits[0]
        probs = logits.softmax(-1)
        top_p, top_id = probs.max(-1)
        blank = self.processor.tokenizer.pad_token_id
        keep = top_id != blank
        confidence = float(top_p[keep].mean()) if keep.any() else 0.0
        text = self.processor.batch_decode(top_id.unsqueeze(0))[0]
        return text, confidence
