"""Run one explicit native REALLMS chat-metadata experiment request."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import shlex
import sys
from typing import Any

from ampav.reallms.chat import ReallmsChatCompletions


METADATA_CATEGORIES = "summary, named entities, keyphrases, and labels"
LABEL_DEFINITION = "broad discovery-oriented concepts useful for finding this transcript"


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


def build_prompt(mode: str, text: str, category: str | None, term: str | None) -> str:
    """Build the experiment prompt while retaining provider-native output."""
    if mode == "combined":
        request = (
            f"Return a JSON object with {METADATA_CATEGORIES}. "
            f"Use labels for {LABEL_DEFINITION}."
        )
    elif mode == "focused":
        if not category:
            raise ValueError("focused mode requires --category")
        request = f"Return a JSON object containing only {category}."
    elif mode == "terminology":
        if not term:
            raise ValueError("terminology mode requires --term")
        request = (
            f"Return a JSON object with a labels array of up to 10 {term}. "
            f"Interpret {term} as {LABEL_DEFINITION}."
        )
    elif mode == "terminology-diagnostic":
        request = (
            "Return a JSON object with separate subjects and topics arrays, each with up to 10 labels. "
            f"Interpret both as {LABEL_DEFINITION}, but keep arrays separate."
        )
    else:
        raise ValueError(f"unsupported mode: {mode}")
    return f"{request}\n\nTranscript:\n{text}"


def sanitize_response(response: dict[str, Any]) -> dict[str, Any]:
    """Remove provider reasoning while retaining the native result structure.

    REALLMS can return internal reasoning beside the requested completion.  It
    is not needed to evaluate metadata behavior and is not retained in a run
    record.
    """
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


def describe_completion_format(response: dict[str, Any]) -> str:
    """Describe whether the first native completion content is parseable JSON."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return "missing first choice"
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        return "missing string message content"
    try:
        parsed = json.loads(message["content"])
    except json.JSONDecodeError:
        return "not strict JSON"
    return "strict JSON object" if isinstance(parsed, dict) else "strict JSON, but not an object"


def has_provider_reasoning(response: dict[str, Any]) -> bool:
    """Report whether a native response exposed provider reasoning fields."""
    choices = response.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        if "reasoning_content" in message:
            return True
        provider_fields = message.get("provider_specific_fields")
        if isinstance(provider_fields, dict) and "reasoning" in provider_fields:
            return True
    return False


def write_run_record(
    output_dir: Path,
    arguments: argparse.Namespace,
    native_response: dict[str, Any],
) -> None:
    """Persist a selected native response and concise, credential-free metadata."""
    output_dir.mkdir(parents=True, exist_ok=False)
    response = sanitize_response(native_response)
    manifest = {
        "timestamp": datetime.now(UTC).isoformat(),
        "api": "REALLMS /chat/completions",
        "model": arguments.model,
        "fixture_id": arguments.fixture_id,
        "mode": arguments.mode,
        "category": arguments.category,
        "term": arguments.term,
        "temperature": arguments.temperature,
        "timeout_seconds": arguments.timeout,
        "python_version": platform.python_version(),
    }
    (output_dir / "manifest.yaml").write_text(
        "\n".join(
            f"{name}: {json.dumps(value)}" for name, value in manifest.items()
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "native_output.json").write_text(
        json.dumps(response, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "command.txt").write_text(
        shlex.join([sys.executable, *sys.argv]) + "\n", encoding="utf-8"
    )
    (output_dir / "input_ref.txt").write_text(
        f"fixture_id: {arguments.fixture_id}\ntext_path: {arguments.text_path}\n"
        "cleanup: no caller-owned inputs or remote resources were created\n",
        encoding="utf-8",
    )
    choices = response.get("choices")
    first_choice = choices[0] if isinstance(choices, list) and choices else {}
    finish_reason = first_choice.get("finish_reason") if isinstance(first_choice, dict) else None
    (output_dir / "observations.md").write_text(
        "# Observations\n\n"
        f"- Native finish reason: `{finish_reason}`.\n"
        f"- Completion content: {describe_completion_format(response)}.\n"
        f"- Provider reasoning removed from retained response: {'yes' if has_provider_reasoning(native_response) else 'no'}.\n"
        "- Metadata quality: pending direct review.\n",
        encoding="utf-8",
    )


def parse_arguments() -> argparse.Namespace:
    """Parse an explicit one-request native experiment invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--text-path", type=Path, required=True)
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("combined", "focused", "terminology", "terminology-diagnostic"), required=True)
    parser.add_argument("--category")
    parser.add_argument("--term", choices=("subjects", "topics", "themes", "categories"))
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    """Call REALLMS once and save the native response in the requested run directory."""
    arguments = parse_arguments()
    load_environment_file(arguments.env_file)
    base_url = os.environ.get("REALLMS_BASE_URL")
    api_key = os.environ.get("REALLMS_API_KEY")
    if not base_url or not api_key:
        raise RuntimeError("REALLMS_BASE_URL and REALLMS_API_KEY must be set")

    text = arguments.text_path.read_text(encoding="utf-8")
    prompt = build_prompt(arguments.mode, text, arguments.category, arguments.term)
    client = ReallmsChatCompletions(base_url, api_key, timeout=arguments.timeout)
    response = client.process(
        arguments.model,
        [
            {"role": "system", "content": "Return JSON only. Do not add markdown fences."},
            {"role": "user", "content": prompt},
        ],
        temperature=arguments.temperature,
    )
    write_run_record(arguments.output_dir, arguments, response)
    print(arguments.output_dir)


if __name__ == "__main__":
    main()
