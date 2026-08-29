"""Run one explicit native REALLMS completions metadata experiment request."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ampav.reallms import ReallmsCompletions

try:
    from experiments.chat_metadata_probe import (
        build_prompt,
        load_environment_file,
        write_run_record,
    )
except ModuleNotFoundError:
    from chat_metadata_probe import build_prompt, load_environment_file, write_run_record


def parse_arguments() -> argparse.Namespace:
    """Parse an explicit native ``/completions`` experiment invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--text-path", type=Path, required=True)
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    """Call REALLMS ``/completions`` once and save the native response."""
    arguments = parse_arguments()
    arguments.api = "REALLMS /completions"
    arguments.mode = "combined"
    load_environment_file(arguments.env_file)
    base_url = os.environ.get("REALLMS_BASE_URL")
    api_key = os.environ.get("REALLMS_API_KEY")
    if not base_url or not api_key:
        raise RuntimeError("REALLMS_BASE_URL and REALLMS_API_KEY must be set")

    text = arguments.text_path.read_text(encoding="utf-8")
    prompt = build_prompt("combined", text, None, None)
    client = ReallmsCompletions(base_url, api_key, timeout=arguments.timeout)
    response = client.process(arguments.model, prompt, temperature=arguments.temperature)
    write_run_record(arguments.output_dir, arguments, response)
    print(arguments.output_dir)


if __name__ == "__main__":
    main()
