"""Load gold annotations and MeetingBank transcripts."""
import json
import logging
from collections import defaultdict
from pathlib import Path

from .schema import Commitment

log = logging.getLogger(__name__)


def _read_records(path):
    text = Path(path).read_text(encoding="utf-8").strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        # Fall back to JSON Lines (one record per line).
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_gold(path):
    """Return {uid: [Commitment, ...]}."""
    by_uid = defaultdict(list)
    for i, raw in enumerate(_read_records(path)):
        rec = Commitment.model_validate(raw)
        if not rec.uid:
            raise ValueError(f"Gold record {i} has no uid")
        by_uid[rec.uid].append(rec)
    return dict(by_uid)


def load_transcripts(uids, cfg):
    """Return {uid: transcript} for the requested uids.

    Uses a local JSON file ({uid: transcript}) if cfg['transcripts_path'] is
    set, otherwise pulls from the Hugging Face MeetingBank release and caches
    the needed transcripts locally.
    """
    uids = set(uids)
    if cfg.get("transcripts_path"):
        data = json.loads(Path(cfg["transcripts_path"]).read_text(encoding="utf-8"))
        missing = uids - data.keys()
        if missing:
            raise KeyError(f"Transcripts missing for: {sorted(missing)}")
        return {u: data[u] for u in uids}

    cache_path = Path(cfg.get("transcript_cache", "data/transcripts_cache.json"))
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    missing = uids - cache.keys()
    if missing:
        from datasets import load_dataset  # imported lazily; only needed here

        ds = load_dataset(cfg.get("hf_dataset", "huuuyeah/meetingbank"))
        for split in ds:
            for row in ds[split]:
                if row["uid"] in missing:
                    cache[row["uid"]] = row["transcript"]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
        still_missing = uids - cache.keys()
        if still_missing:
            raise KeyError(f"uids not found in MeetingBank: {sorted(still_missing)}")
    return {u: cache[u] for u in uids}
