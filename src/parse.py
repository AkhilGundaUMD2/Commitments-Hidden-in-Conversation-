"""Turn raw model text into validated Commitment records."""
import json
import re

from pydantic import ValidationError

from .schema import Commitment


class ParseError(Exception):
    pass


def extract_json_array(text):
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        # Some models wrap the array in an object.
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            raise ParseError("no JSON array found") from e
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, list):
                    return v
        raise ParseError("no JSON array found")
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise ParseError(f"invalid JSON: {e}") from e
    if not isinstance(data, list):
        raise ParseError("top-level JSON is not an array")
    return data


def parse_records(text, uid):
    """Return (valid_records, invalid_items_with_errors)."""
    valid, invalid = [], []
    for item in extract_json_array(text):
        try:
            rec = Commitment.model_validate(item)
            rec.uid = uid
            valid.append(rec)
        except ValidationError as e:
            invalid.append({"item": item, "error": str(e)})
    return valid, invalid
