"""Speech recognition through REALLMS."""

from __future__ import annotations

import base64
from typing import Any

from ._chat import ReallmsChatClient


DEFAULT_ASR_PROMPT = (
    "Transcribe the speech in this audio accurately. Preserve the spoken wording "
    "and do not summarize or add commentary."
)


class ReallmsAsr:
    """Transcribe inline audio with a REALLMS speech-recognition model."""

    def __init__(self, base_url: str, api_key: str, *, timeout: float = 60.0) -> None:
        self._client = ReallmsChatClient(base_url, api_key, timeout=timeout)

    def process(
        self,
        audio: bytes,
        *,
        media_type: str,
        model: str,
        prompt: str = DEFAULT_ASR_PROMPT,
        **request_options: Any,
    ) -> dict[str, Any]:
        """Transcribe audio and return the decoded native response.

        ``media_type`` is the audio MIME type used in the inline data URL.
        Additional keyword arguments are passed through as native request fields.
        """
        if not audio:
            raise ValueError("audio must not be empty")
        if not media_type.startswith("audio/"):
            raise ValueError("media_type must be an audio MIME type")
        if not prompt:
            raise ValueError("prompt must not be empty")

        encoded_audio = base64.b64encode(audio).decode("ascii")
        return self._client.complete(
            model,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "audio_url",
                            "audio_url": {
                                "url": f"data:{media_type};base64,{encoded_audio}",
                            },
                        },
                    ],
                },
            ],
            **request_options,
        )
