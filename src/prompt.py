"""Direct-prompting baseline prompt.

TODO: replace the label definitions below with the exact wording from your
annotation guide so the model and annotators share one definition.
"""
import json

PROMPT_VERSION = "direct_v1"

EXAMPLE = {
    "commitment_id": "C001",
    "owner": "Emily Armstrong and Art Sanchez",
    "action": "amend",
    "object": "ordinance",
    "commitment_force": "firm_commitment",
    "condition": None,
    "deadline": None,
    "status": "active",
    "evidence": {"text": "council at the May 10th meeting, directed staff to amend the ordinance"},
    "revision_of": None,
    "uncertainty": False,
}

SYSTEM = """You extract commitments from meeting transcripts.

A commitment is any statement that a person or group will, should, or may do something.
For each one, output a record with these fields:
- commitment_id: "C001", "C002", ... in order of first appearance
- owner: person or group responsible, or null if unstated
- action: the verb (e.g. "amend", "report back")
- object: what the action applies to, or null
- commitment_force: one of
    "suggestion" (an action is proposed but nobody accepts responsibility),
    "intention" (the speaker states a plan without promising it to anyone),
    "conditional_promise" (responsibility is accepted only if a condition holds),
    "firm_commitment" (responsibility is accepted or assigned without conditions)
- condition: the prerequisite for a conditional_promise, else null
- deadline: the due date or time as stated, else null
- status: one of "proposed", "active", "revised", "completed", "cancelled", "uncertain"
  (the status at the END of the transcript)
- evidence: {"text": "..."} copied VERBATIM from the transcript, the shortest span that supports the record
- revision_of: commitment_id of the earlier record this one revises, else null
- uncertainty: true if the transcript supports more than one reasonable interpretation

Rules:
- Only extract commitments supported by the transcript. Do not infer unstated ones.
- Evidence must be an exact quote. Never paraphrase it.
- Return ONLY a JSON array of records. Return [] if there are none."""


def build_messages(transcript):
    user = (
        "Example record:\n"
        f"{json.dumps(EXAMPLE, indent=2)}\n\n"
        "Transcript:\n"
        f"{transcript}\n\n"
        "Return the JSON array of commitments."
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


RETRY_MESSAGE = (
    "Your previous response was not a valid JSON array. "
    "Return only the JSON array of commitment records, with no other text."
)
