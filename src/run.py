"""Run the direct-prompting baseline end to end.

    python -m src.run --config config.yaml
    python -m src.run --config config.yaml --mock      # no API calls
"""
import argparse
import json
import logging
import re
import time
from pathlib import Path

import yaml

from .load import load_gold, load_transcripts
from .match import grounding_score, match, norm
from .metrics import aggregate, score_pair
from .model import get_model
from .parse import ParseError, parse_records
from .prompt import PROMPT_VERSION, RETRY_MESSAGE, build_messages

log = logging.getLogger("baseline")


def run_one(uid, transcript, golds, model, cfg, raw_dir):
    messages = build_messages(transcript)
    text = model(uid, messages)
    (raw_dir / f"{uid}.txt").write_text(text, encoding="utf-8")

    parse_failed = False
    try:
        preds, invalid = parse_records(text, uid)
    except ParseError:
        log.warning("%s: parse failed, retrying once", uid)
        retry = messages + [{"role": "assistant", "content": text},
                            {"role": "user", "content": RETRY_MESSAGE}]
        text = model(uid, retry)
        (raw_dir / f"{uid}.retry.txt").write_text(text, encoding="utf-8")
        try:
            preds, invalid = parse_records(text, uid)
        except ParseError as e:
            log.error("%s: parse failed after retry: %s", uid, e)
            preds, invalid, parse_failed = [], [], True

    t_norm = norm(transcript)
    grounding = [grounding_score(p.evidence.text, t_norm) for p in preds]
    grounded = [s >= cfg.get("grounding_threshold", 90) for s in grounding]

    pairs = match(preds, golds, cfg.get("match_threshold", 0.6), cfg.get("evidence_weight", 0.6))
    pair_scores = [score_pair(preds[pi], golds[gi], cfg.get("text_threshold", 80)) for pi, gi, _ in pairs]

    detail = {
        "uid": uid,
        "predictions": [
            {**p.model_dump(), "grounding_score": s, "grounded": g}
            for p, s, g in zip(preds, grounding, grounded)
        ],
        "invalid_records": invalid,
        "matches": [
            {"pred_id": preds[pi].commitment_id, "gold_id": golds[gi].commitment_id,
             "score": sc, "fields": fs}
            for (pi, gi, sc), fs in zip(pairs, pair_scores)
        ],
        "unmatched_pred": [p.commitment_id for i, p in enumerate(preds) if i not in {pi for pi, _, _ in pairs}],
        "unmatched_gold": [g.commitment_id for i, g in enumerate(golds) if i not in {gi for _, gi, _ in pairs}],
    }
    summary = {
        "uid": uid,
        "n_pred": len(preds),
        "n_gold": len(golds),
        "n_matched": len(pairs),
        "n_ungrounded": sum(not g for g in grounded),
        "n_invalid": len(invalid),
        "parse_failed": parse_failed,
        "pair_scores": pair_scores,
    }
    return summary, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--mock", action="store_true", help="use the mock model (no API calls)")
    ap.add_argument("--limit", type=int, help="only run the first N meetings")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.mock:
        cfg["mock"] = True

    gold = load_gold(cfg["gold_path"])
    uids = sorted(gold)[: args.limit] if args.limit else sorted(gold)
    transcripts = load_transcripts(uids, cfg)
    model = get_model(cfg, gold)

    name = "mock" if cfg.get("mock") else re.sub(r"[^\w.-]", "_", cfg["model"])
    out_dir = Path(cfg.get("runs_dir", "runs")) / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True)
    (out_dir / "config.json").write_text(json.dumps({**cfg, "prompt_version": PROMPT_VERSION}, indent=2))

    summaries, details = [], []
    for uid in uids:
        log.info("running %s", uid)
        s, d = run_one(uid, transcripts[uid], gold[uid], model, cfg, raw_dir)
        summaries.append(s)
        details.append(d)

    metrics = aggregate(summaries)
    (out_dir / "details.json").write_text(json.dumps(details, indent=2))
    (out_dir / "per_meeting.json").write_text(json.dumps(
        [{k: v for k, v in s.items() if k != "pair_scores"} for s in summaries], indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nResults written to {out_dir}\n")
    for k, v in metrics.items():
        print(f"  {k:24s} {v:.3f}" if isinstance(v, float) else f"  {k:24s} {v}")


if __name__ == "__main__":
    main()
