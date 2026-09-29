"""Command-line entry point for REALLMS speech recognition."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import logging
import os
from pathlib import Path

from ampav.core.async_tool import ToolError
from ampav.core.logging import LOG_FORMAT
from ampav.reallms.asr import DEFAULT_ASR_PROMPT
from ampav_reallms_pipeline import transcribe_file


def build_cli_parser() -> argparse.ArgumentParser:
    """Build the REALLMS ASR CLI parser."""
    parser = argparse.ArgumentParser(
        description="Transcribe local audio with REALLMS and print AMPAV ToolOutput YAML."
    )
    parser.add_argument("media", type=Path, help="Local audio file")
    parser.add_argument(
        "--base-url",
        required=True,
        help="REALLMS API base URL, without the endpoint path",
    )
    parser.add_argument("--model", default="Qwen3-ASR-1.7B", help="REALLMS ASR model")
    parser.add_argument("--prompt", default=DEFAULT_ASR_PROMPT, help="Transcription prompt")
    parser.add_argument(
        "--media-type",
        help="Audio MIME type; inferred from the filename when omitted",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Native sampling temperature",
    )
    parser.add_argument(
        "--include-tool-private",
        action="store_true",
        help="Include the decoded native response in ToolOutput.tool_private",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="Request timeout in seconds")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the REALLMS ASR CLI."""
    args = build_cli_parser().parse_args(argv)
    logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG if args.debug else logging.INFO)

    try:
        api_key = os.environ.get("REALLMS_API_KEY")
        if not api_key:
            raise ValueError("REALLMS_API_KEY must be set")
        result = transcribe_file(
            args.media,
            base_url=args.base_url,
            api_key=api_key,
            model=args.model,
            prompt=args.prompt,
            media_type=args.media_type,
            include_tool_private=args.include_tool_private,
            timeout=args.timeout,
            temperature=args.temperature,
        )
    except Exception as exc:
        cli_errors = (ToolError, OSError, ValueError)
        if not isinstance(exc, cli_errors):
            raise
        logging.error("%s", exc)
        return 1

    print(result.model_dump_yaml(sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
