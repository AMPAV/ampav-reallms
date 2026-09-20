"""Run and retain one native REALLMS speech-recognition request."""

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
from typing import Any

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


def sanitize_response(response: dict[str, Any]) -> dict[str, Any]:
    """Remove provider reasoning while retaining normal native response data."""
    sanitized = json.loads(json.dumps(response))
    for choice in sanitized.get("choices", []):
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict):
            message.pop("reasoning_content", None)
            provider_fields = message.get("provider_specific_fields")
            if isinstance(provider_fields, dict):
                provider_fields.pop("reasoning", None)
    return sanitized


def completion_content(response: dict[str, Any]) -> str | None:
    """Return the first completion's string content when present."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    return content if isinstance(content, str) else None


def write_run_record(
    output_dir: Path,
    arguments: argparse.Namespace,
    media_type: str,
    input_bytes: int,
    started: datetime,
    elapsed_seconds: float,
    native_response: dict[str, Any],
) -> None:
    """Persist selected native output and credential-free run metadata."""
    output_dir.mkdir(parents=True, exist_ok=False)
    response = sanitize_response(native_response)
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
    (output_dir / "native_output.json").write_text(
        json.dumps(response, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
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
    choices = response.get("choices")
    first_choice = choices[0] if isinstance(choices, list) and choices else {}
    finish_reason = first_choice.get("finish_reason") if isinstance(first_choice, dict) else None
    content = completion_content(response)
    (output_dir / "observations.md").write_text(
        "# Observations\n\n"
        f"- Native finish reason: `{finish_reason}`.\n"
        f"- String transcript content returned: {'yes' if content else 'no'}.\n"
        "- Review wording, omissions, insertions, punctuation, and native timing or language metadata.\n"
        "- Adoption assessment: pending direct review.\n",
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
    tool = ReallmsAsr(base_url, api_key, timeout=arguments.timeout)
    response = tool.process(
        audio,
        media_type=media_type,
        model=arguments.model,
        temperature=arguments.temperature,
    )
    write_run_record(
        arguments.output_dir,
        arguments,
        media_type,
        len(audio),
        started,
        monotonic() - elapsed_started,
        response,
    )
    print(arguments.output_dir)


if __name__ == "__main__":
    main()
