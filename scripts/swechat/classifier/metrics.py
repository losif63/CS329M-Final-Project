"""Evaluation metrics for the slop classifier."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """The score cutoff that maximizes F1. Chosen on val, then applied unchanged to test."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])


def evaluate(y_true: np.ndarray, predicted: np.ndarray) -> dict:
    """Precision, recall and F1 for the slop class, given yes/no predictions.

    Precision is None when nothing is predicted as slop (0/0 is undefined).
    """
    flagged = int(predicted.sum())
    correct = int((predicted & (y_true == 1)).sum())
    precision = correct / flagged if flagged else None
    recall = correct / int(y_true.sum())
    f1 = 2 * precision * recall / (precision + recall) if precision else 0.0
    return {
        "sentences": len(y_true),
        "slop": int(y_true.sum()),
        "flagged": flagged,
        "correct": correct,
        "precision": None if precision is None else round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }
