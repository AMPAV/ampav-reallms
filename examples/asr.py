"""Transcribe the bundled MP3 file directly with the REALLMS ASR tool.

Set ``REALLMS_API_KEY`` before running this example.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ampav.reallms import ReallmsAsr


INPUT_FILE = Path(__file__).parent / "data" / "OpenDoor.mp3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("--model", default="Qwen3-ASR-1.7B")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    api_key = os.environ.get("REALLMS_API_KEY")
    if not api_key:
        raise RuntimeError("REALLMS_API_KEY must be set")

    tool = ReallmsAsr(
        args.base_url,
        api_key,
        model=args.model,
        timeout=args.timeout,
    )
    output = tool.process(
        INPUT_FILE.read_bytes(),
        media_type="audio/mpeg",
        temperature=args.temperature,
    )
    print(output.model_dump_yaml(sort_keys=False))


if __name__ == "__main__":
    main()
