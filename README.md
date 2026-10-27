# Direct-prompting baseline

## Setup
```
pip install -r requirements.txt
export OPENAI_API_KEY=...   # or whichever provider the model uses
```

## Run
```
python -m src.run --config config.yaml --mock   # smoke test; no API calls
python -m src.run --config config.yaml          # real run
python -m src.run --config config.yaml --limit 2
```
Mock mode with the bundled sample transcript:
```
python -m src.run --config config.yaml --mock   # after setting transcripts_path: data/sample_transcripts.json
```

## Outputs (`runs/<model>_<timestamp>/`)
- `metrics.json`: aggregate metrics
- `per_meeting.json`: counts per meeting
- `details.json`: every prediction, grounding score, match, and field result (use for error analysis)
- `raw/`: raw model responses
- `config.json`: config and prompt version used

## Metrics
- Commitment precision / recall / F1 (one-to-one matching on evidence + action/object similarity)
- Field accuracy on matched pairs: force, status, uncertainty (exact); action, object, condition, deadline (fuzzy); owner (set F1)
- Unsupported rate: predictions whose evidence is not found in the transcript
- Parse failures and invalid records

## Changing models
Edit `model` in `config.yaml`. Any LiteLLM model string works.
