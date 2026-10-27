"""Evidence grounding and prediction-to-gold matching."""
import re

import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment


def norm(s):
    s = (s or "").lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def grounding_score(evidence, transcript_norm):
    """0-100 score for how well the evidence appears in the transcript."""
    e = norm(evidence)
    if not e:
        return 0.0
    if e in transcript_norm:
        return 100.0
    return fuzz.partial_ratio(e, transcript_norm)


def pair_score(pred, gold, evidence_weight=0.6):
    ev = fuzz.partial_ratio(norm(pred.evidence.text), norm(gold.evidence.text)) / 100
    ao = fuzz.token_set_ratio(
        norm(f"{pred.action} {pred.object or ''}"),
        norm(f"{gold.action} {gold.object or ''}"),
    ) / 100
    return evidence_weight * ev + (1 - evidence_weight) * ao


def match(preds, golds, threshold=0.6, evidence_weight=0.6):
    """One-to-one matching. Returns [(pred_idx, gold_idx, score), ...]."""
    if not preds or not golds:
        return []
    m = np.array([[pair_score(p, g, evidence_weight) for g in golds] for p in preds])
    rows, cols = linear_sum_assignment(-m)
    return [(int(r), int(c), float(m[r, c])) for r, c in zip(rows, cols) if m[r, c] >= threshold]
