"""Model-agnostic API layer (LiteLLM) plus a mock model for testing."""
import json


def get_model(cfg, gold_by_uid=None):
    """Return a callable (uid, messages) -> response text."""
    if cfg.get("mock"):
        return MockModel(gold_by_uid or {})

    import litellm  # imported lazily so mock runs need no API packages

    def call(uid, messages):
        resp = litellm.completion(
            model=cfg["model"],
            messages=messages,
            temperature=cfg.get("temperature", 0),
            max_tokens=cfg.get("max_tokens", 8000),
            **cfg.get("model_kwargs", {}),
        )
        return resp.choices[0].message.content or ""

    return call


class MockModel:
    """Echoes gold with one perturbation and one invented record.

    Lets you check parsing, grounding, matching, and metrics without an API key.
    """

    def __init__(self, gold_by_uid):
        self.gold = gold_by_uid

    def __call__(self, uid, messages):
        out = []
        for i, rec in enumerate(self.gold.get(uid, [])):
            d = rec.model_dump(exclude={"uid"})
            d["commitment_id"] = f"P{i + 1:03d}"
            if i == 0 and d.get("owner") and " and " in d["owner"]:
                d["owner"] = d["owner"].split(" and ")[0]  # partial owner
            out.append(d)
        out.append({
            "commitment_id": f"P{len(out) + 1:03d}",
            "owner": "staff",
            "action": "schedule",
            "object": "follow-up hearing",
            "commitment_force": "intention",
            "condition": None,
            "deadline": "next month",
            "status": "proposed",
            "evidence": {"text": "this sentence does not appear in the transcript"},
            "revision_of": None,
            "uncertainty": False,
        })
        # Wrapped in a code fence to exercise the parser.
        return "```json\n" + json.dumps(out, indent=2) + "\n```"
