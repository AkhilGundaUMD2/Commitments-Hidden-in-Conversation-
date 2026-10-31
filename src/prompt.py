"""Direct-prompting baseline prompt.

Label definitions and rules follow the D1 Annotation Guide so the model and
annotators share one definition.
"""
import json

PROMPT_VERSION = "direct_v2"

EXAMPLE = {
    "commitment_id": "C001",
    "version": 1,
    "owner": "Speaker A",
    "action": "finish",
    "object": "report",
    "commitment_force": "firm_commitment",
    "conditions_cnf": [],
    "deadline": "Friday",
    "status": "active",
    "evidence": [{"source": "transcript", "text": "I'll finish the report by Friday."}],
    "revision_of": None,
    "uncertainty": False,
}

SYSTEM = """You extract commitments from meeting transcripts.

Label statements about a task someone may do in the future: suggestions, plans,
conditional promises, and firm promises. Also track later changes to earlier commitments.

Do NOT label:
- Requests nobody accepted ("Can someone write a report?", "May we have a staff report?").
  If someone replies "I'll write it", that reply is a firm commitment.
- General meeting instructions ("Please turn on your microphone", "Please use the raise hand feature").
- Statements of ability ("I can write code") unless the person offers to do a specific task.
- Hearsay about others ("I think John is working on it") as proof that they accepted the task.
A clear task suggestion may be kept as a suggestion with status "proposed".

For each commitment, output a record with these fields:
- commitment_id: "C001", "C002", ... in order of first appearance. Each separate task gets its own ID.
- version: 1 for the first version of a commitment; increase it for each revision.
- owner: who agreed to do the task (one person or a group). Use "unknown" if it cannot be
  determined, and set uncertainty to true.
- action: what they will do, e.g. "review", "send", "prepare".
- object: what the action is about, e.g. "report", "dataset"; null if not stated.
- commitment_force: how strongly someone agrees to do the task. One of
    "suggestion" (someone proposes a task but does not agree to do it, e.g. "We should meet tomorrow."),
    "intention" (someone has a plan but makes no clear promise, e.g. "I plan to finish the report."),
    "conditional_promise" (someone agrees to do a task if a condition is met, e.g. "I'll finish it if I get the data."),
    "firm_commitment" (someone clearly agrees to do a task, e.g. "I'll send it on Friday.")
- conditions_cnf: conditions that must hold before the promise applies, in Conjunctive Normal Form:
  a list of groups joined by AND, where each group is a list of choices joined by OR.
    "if the dataset is ready and the baseline is finished" -> [["dataset ready"], ["baseline finished"]]
    "if the main or backup dataset is ready, and the baseline is finished"
      -> [["main dataset ready", "backup dataset ready"], ["baseline finished"]]
  Use [] if no condition is stated. Keep the original meaning.
- deadline: when the task should be done, keeping the date words as they appear in the transcript;
  null if no deadline was given. Never make one up. A date inside a condition is not the deadline
  ("I'll test it if the files arrive by Friday": Friday belongs to the condition).
- status: what has happened to the task by the END of the transcript. One of
    "proposed" (a task was suggested, but no one accepted it),
    "active" (a plan or promise is still in effect),
    "revised" (this version was later changed by a newer version),
    "completed" (someone says the work was finished, or there is other clear evidence of completion),
    "cancelled" (the commitment was withdrawn),
    "uncertain" (the current status cannot be told from the text)
- evidence: a list of spans, each {"source": "transcript", "text": "..."}.
- revision_of: for a revised version, the earlier version it replaces as "<commitment_id>-v<version>"
  (e.g. "C001-v1"); otherwise null.
- uncertainty: true if an important detail is unclear or the text supports more than one interpretation.

Force rules:
- Read the nearby conversation before choosing a label.
- Do not assume a person named in a suggestion accepted the task.
- Hedged statements ("I might finish tomorrow") are not firm promises; they may be intentions.
- "I recommend a new policy" is a suggestion, not a promise to make the policy.
- In city council meetings, a direction to staff does not always prove a named staff member
  accepted the task. Look for evidence of responsibility before naming an owner.
  A public comment asking for a change is not an active promise by the council.

Status rules:
- Do not mark a task completed only because its deadline passed.
- A suggestion nobody accepted is normally "proposed", not "active".
- A conditional promise can still be "active" while the condition is pending.
- Do not assume a condition was met unless the transcript provides evidence.

Evidence rules:
- Copy the words exactly. Never paraphrase or rewrite the quote.
- Use the shortest span that still proves the label.
- If the same words appear more than once, include enough nearby text to locate the right place.
- Use more than one span if one quote does not support every detail (e.g. one span naming the
  task and another showing who accepted it).

Revisions:
- A task can change its owner, action, condition, or deadline. Keep the old version and add a new
  record with the SAME commitment_id, the next version number, and revision_of pointing to the old
  version. Mark the old version "revised". Include the text that explains the change as evidence.
- Only link records when the conversation shows they are about the same task, not just the same topic.
- A progress update ("I've started the report") is not a revision unless a recorded field or the
  status changes.
- If one statement contains two separate tasks ("I'll send the files and write the summary"),
  make two records.

Do not guess. Only extract commitments supported by the transcript.
Return ONLY a JSON array of records. Return [] if there are none."""


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
