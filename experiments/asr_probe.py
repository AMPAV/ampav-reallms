"""Run and retain one REALLMS speech-recognition request."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import mimetypes
import os
from pathlib import Path
import platform
import shlex
import sys
from time import monotonic

from ampav.core.schema import ToolOutput
from ampav.reallms import ReallmsAsr


def load_environment_file(path: Path) -> None:
    """Load simple ``NAME=value`` entries for this local experiment client."""
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name, separator, value = stripped.partition("=")
        if not separator or not name.isidentifier():
            raise ValueError(f"invalid environment entry in {path}")
        os.environ.setdefault(name, value.strip().strip('"').strip("'"))


def write_run_record(
    output_dir: Path,
    arguments: argparse.Namespace,
    media_type: str,
    input_bytes: int,
    started: datetime,
    elapsed_seconds: float,
    tool_output: ToolOutput,
) -> None:
    """Persist normalized output and credential-free run metadata."""
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "timestamp": started.isoformat(),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "api": "REALLMS /chat/completions",
        "model": arguments.model,
        "fixture_id": arguments.fixture_id,
        "prompt": "default verbatim transcription instruction",
        "input_media_type": media_type,
        "input_bytes": input_bytes,
        "temperature": arguments.temperature,
        "timeout_seconds": arguments.timeout,
        "python_version": platform.python_version(),
    }
    (output_dir / "manifest.yaml").write_text(
        "\n".join(f"{name}: {json.dumps(value)}" for name, value in manifest.items()) + "\n",
        encoding="utf-8",
    )
    (output_dir / "tool_output.yaml").write_text(
        tool_output.model_dump_yaml(sort_keys=False), encoding="utf-8"
    )
    (output_dir / "command.txt").write_text(
        shlex.join([sys.executable, *sys.argv]) + "\n", encoding="utf-8"
    )
    (output_dir / "input_ref.txt").write_text(
        f"fixture_id: {arguments.fixture_id}\naudio_path: {arguments.audio_path}\n"
        "provider_input: inline audio data URL\n"
        "cleanup: no caller-owned inputs or remote resources were created\n",
        encoding="utf-8",
    )
    transcript = tool_output.output
    (output_dir / "observations.md").write_text(
        "# Observations\n\n"
        f"- Transcript text returned: {'yes' if transcript and transcript.text else 'no'}.\n"
        "- REALLMS did not return word timing, diarization, confidence, or paragraph data.\n"
        "- Review wording, omissions, insertions, and punctuation.\n",
        encoding="utf-8",
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--audio-path", type=Path, required=True)
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen3-ASR-1.7B")
    parser.add_argument("--media-type")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    load_environment_file(arguments.env_file)
    base_url = os.environ.get("REALLMS_BASE_URL")
    api_key = os.environ.get("REALLMS_API_KEY")
    if not base_url or not api_key:
        raise RuntimeError("REALLMS_BASE_URL and REALLMS_API_KEY must be set")

    audio = arguments.audio_path.read_bytes()
    media_type = arguments.media_type or mimetypes.guess_type(arguments.audio_path.name)[0]
    if not media_type:
        raise ValueError("could not determine audio media type; pass --media-type")
    started = datetime.now(UTC)
    elapsed_started = monotonic()
    tool = ReallmsAsr(
        base_url,
        api_key,
        model=arguments.model,
        timeout=arguments.timeout,
    )
    output = tool.process(
        audio,
        media_type=media_type,
        temperature=arguments.temperature,
    )
    write_run_record(
        arguments.output_dir,
        arguments,
        media_type,
        len(audio),
        started,
        monotonic() - elapsed_started,
        output,
    )
    print(arguments.output_dir)


if __name__ == "__main__":
    main()
