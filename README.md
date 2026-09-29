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

### Local-file pipeline

`ampav_reallms_pipeline.transcribe_file` reads caller-owned local media,
infers its MIME type when possible, and dispatches the same synchronous tool.
It never modifies or deletes the source file.

```python
from ampav_reallms_pipeline import transcribe_file

result = transcribe_file(
    "speech.mp3",
    base_url="https://reallms.example/v1",
    api_key="api-key",
    model="Qwen3-ASR-1.7B",
    temperature=0,
)
print(result.output.text)
```

Pass `media_type` explicitly when it cannot be inferred from the filename.

### Command line

The CLI reads the API key from `REALLMS_API_KEY` so it is not placed in shell
history or process arguments:

```bash
export REALLMS_API_KEY="..."
ampav_reallms_asr speech.mp3 \
  --base-url https://reallms.example/v1 \
  --model Qwen3-ASR-1.7B
```

It prints the complete AMPAV `ToolOutput` as YAML. Use
`--include-tool-private` only for provider troubleshooting.

### Examples

The repository includes a small MP3 fixture and direct-tool and file-pipeline
examples:

```bash
python examples/asr.py https://reallms.example/v1
python examples/asr_file.py https://reallms.example/v1
```

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
