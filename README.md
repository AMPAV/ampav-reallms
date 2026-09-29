# ampav-reallms

REALLMS tooling for the AMPAV environment.

The public API is under development.

## Speech recognition

`ReallmsAsr` sends an inline audio data URL to the synchronous REALLMS
`/chat/completions` endpoint and returns `ToolOutput.output = Transcript`.
REALLMS currently provides transcript text without word timing, diarization,
confidence, or paragraph data.

```python
from pathlib import Path

from ampav.reallms import ReallmsAsr

tool = ReallmsAsr(
    "https://reallms.example/v1",
    "api-key",
    model="Qwen3-ASR-1.7B",
)
result = tool.process(
    Path("speech.flac").read_bytes(),
    media_type="audio/flac",
    temperature=0,
)
print(result.output.text)
```

Configure the model and transcription prompt when constructing the tool. Pass
request-specific native options to `process`. Set `include_tool_private=True`
only when the decoded native response is needed for troubleshooting.

## Development

Use the shared AMPAV virtual environment and install the package in editable
mode:

```bash
python -m pip install -e .
```

Run the unit tests:

```bash
python -m unittest discover -s tests
```
