"""Local-file pipeline adapter for REALLMS speech recognition."""

from __future__ import annotations

import mimetypes
from os import PathLike
from pathlib import Path
from typing import Any

from ampav.core.schema import ToolOutput
from ampav.reallms import ReallmsAsr
from ampav.reallms.asr import DEFAULT_ASR_PROMPT


def transcribe_file(
    source: str | PathLike[str],
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str = DEFAULT_ASR_PROMPT,
    media_type: str | None = None,
    include_tool_private: bool = False,
    timeout: float = 60.0,
    **request_options: Any,
) -> ToolOutput:
    """Read a local audio file and transcribe it synchronously.

    Parameters:
        source: Local caller-owned audio file. The file is only read.
        base_url: REALLMS API base URL, without the endpoint path.
        api_key: API key sent as a bearer token.
        model: REALLMS speech-recognition model identifier.
        prompt: Instruction sent with the audio request.
        media_type: Audio MIME type. When omitted, infer it from the filename.
        include_tool_private: Include the decoded native response for
            troubleshooting.
        timeout: Request timeout in seconds.
        request_options: Additional native request fields.
    """
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {source_path}")
    resolved_media_type = media_type or mimetypes.guess_type(source_path.name)[0]
    if resolved_media_type is None:
        raise ValueError("could not determine audio media type; pass media_type")

    tool = ReallmsAsr(
        base_url,
        api_key,
        model=model,
        prompt=prompt,
        timeout=timeout,
        include_tool_private=include_tool_private,
    )
    return tool.process(
        source_path.read_bytes(),
        media_type=resolved_media_type,
        **request_options,
    )
