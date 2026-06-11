"""Audio content classification using the Audio Spectrogram Transformer (AST).

Uses MIT/ast-finetuned-audioset-10-10-0.4593, a model trained on AudioSet's
527 sound categories (speech, music, traffic, animal sounds, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
TARGET_SR = 16000


@dataclass(frozen=True)
class ClassificationResult:
    labels: list[str]
    scores: list[float]


def load_model():
    """Load the AST feature extractor and model. Call once and cache the result."""
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
    model.eval()
    return extractor, model


def classify_file(
    path: str | Path,
    *,
    extractor,
    model,
    top_k: int = 5,
) -> ClassificationResult:
    """Classify audio content using the Audio Spectrogram Transformer.

    Args:
        path: Path to audio file.
        extractor: HuggingFace feature extractor (from load_model).
        model: HuggingFace classification model (from load_model).
        top_k: Number of top labels to return.

    Returns:
        ClassificationResult with ranked labels and confidence scores.
    """
    import torch

    audio, _ = librosa.load(str(path), sr=TARGET_SR, mono=True)

    inputs = extractor(audio, sampling_rate=TARGET_SR, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    scores = torch.softmax(logits[0], dim=-1).cpu().numpy()
    top_indices = np.argsort(scores)[::-1][:top_k]

    labels = [model.config.id2label[int(i)] for i in top_indices]
    top_scores = [float(scores[i]) for i in top_indices]

    return ClassificationResult(labels=labels, scores=top_scores)
