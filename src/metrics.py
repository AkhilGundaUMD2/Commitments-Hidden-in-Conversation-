"""Field-level and aggregate metrics."""
import re

from rapidfuzz import fuzz

from .match import norm

EXACT_FIELDS = ["commitment_force", "status", "uncertainty"]
TEXT_FIELDS = ["action", "object", "condition", "deadline"]


def split_owners(s):
    parts = re.split(r",|;|&|/|\band\b", (s or "").lower())
    return {norm(p) for p in parts if norm(p)}


def owner_f1(pred, gold):
    p, g = split_owners(pred), split_owners(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    tp = len(p & g)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(p), tp / len(g)
    return 2 * prec * rec / (prec + rec)


def text_match(pred, gold, threshold=80):
    p, g = norm(pred), norm(gold)
    if not p and not g:
        return True
    if not p or not g:
        return False
    return fuzz.token_set_ratio(p, g) >= threshold


def score_pair(pred, gold, text_threshold=80):
    out = {"owner_f1": owner_f1(pred.owner, gold.owner)}
    for f in EXACT_FIELDS:
        out[f] = getattr(pred, f) == getattr(gold, f)
    for f in TEXT_FIELDS:
        out[f] = text_match(getattr(pred, f), getattr(gold, f), text_threshold)
    return out


def prf(tp, n_pred, n_gold):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def aggregate(per_uid):
    """Micro-average over all meetings."""
    n_pred = sum(u["n_pred"] for u in per_uid)
    n_gold = sum(u["n_gold"] for u in per_uid)
    tp = sum(u["n_matched"] for u in per_uid)
    ungrounded = sum(u["n_ungrounded"] for u in per_uid)
    pairs = [s for u in per_uid for s in u["pair_scores"]]

    p, r, f = prf(tp, n_pred, n_gold)
    out = {
        "meetings": len(per_uid),
        "n_pred": n_pred,
        "n_gold": n_gold,
        "n_matched": tp,
        "commitment_precision": p,
        "commitment_recall": r,
        "commitment_f1": f,
        "unsupported_rate": ungrounded / n_pred if n_pred else 0.0,
        "parse_failures": sum(u["parse_failed"] for u in per_uid),
        "invalid_records": sum(u["n_invalid"] for u in per_uid),
    }
    if pairs:
        for k in pairs[0]:
            out[f"{k}_acc" if k != "owner_f1" else k] = sum(float(s[k]) for s in pairs) / len(pairs)
    return out
