"""Speech recognition through REALLMS."""

from __future__ import annotations

import base64
from time import time
from typing import Any

from ampav.core.async_tool import ToolError
from ampav.core.schema import ToolOutput, Transcript

from ._chat import ReallmsChatClient


DEFAULT_ASR_PROMPT = (
    "Transcribe the speech in this audio accurately. Preserve the spoken wording "
    "and do not summarize or add commentary."
)


class ReallmsAsr:
    """Synchronously transcribe inline audio through REALLMS.

    Parameters:
        base_url: REALLMS API base URL, without the endpoint path.
        api_key: API key sent as a bearer token.
        model: REALLMS speech-recognition model identifier.
        prompt: Instruction sent with each audio request.
        timeout: Request timeout in seconds.
        include_tool_private: Include the decoded native response in
            ``ToolOutput.tool_private`` for troubleshooting.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        model: str,
        prompt: str = DEFAULT_ASR_PROMPT,
        timeout: float = 60.0,
        include_tool_private: bool = False,
    ) -> None:
        if not model:
            raise ValueError("model must not be empty")
        if not prompt:
            raise ValueError("prompt must not be empty")
        self._client = ReallmsChatClient(base_url, api_key, timeout=timeout)
        self.model = model
        self.prompt = prompt
        self.include_tool_private = include_tool_private

    def process(
        self,
        audio: bytes,
        *,
        media_type: str,
        **request_options: Any,
    ) -> ToolOutput:
        """Transcribe audio and return an AMPAV transcript.

        ``media_type`` is the audio MIME type used in the inline data URL.
        Additional keyword arguments are passed through as native request fields.

        REALLMS ASR does not return timing, word, paragraph, speaker, or
        confidence data, so the returned ``Transcript`` contains text only.
        """
        if not audio:
            raise ValueError("audio must not be empty")
        if not media_type.startswith("audio/"):
            raise ValueError("media_type must be an audio MIME type")
        reserved_options = {"model", "messages", "prompt"}.intersection(request_options)
        if reserved_options:
            names = ", ".join(sorted(reserved_options))
            raise ValueError(f"request options must not override configured fields: {names}")

        encoded_audio = base64.b64encode(audio).decode("ascii")
        started = time()
        native = self._client.complete(
            self.model,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
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
        ended = time()
        output = ToolOutput(
            tool_name="reallms_asr",
            tool_version=_version(),
            parameters={
                "model": self.model,
                "prompt": self.prompt,
                "media_type": media_type,
                **request_options,
            },
            start_time=started,
            end_time=ended,
            output=_response_to_transcript(native),
        )
        if self.include_tool_private:
            output.tool_private = {"native_response": native}
        return output


def _response_to_transcript(response: dict[str, Any]) -> Transcript:
    """Extract transcript text from a native chat-completion response."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ToolError("REALLMS ASR response did not contain a completion choice")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ToolError("REALLMS ASR response did not contain a completion message")
    content = message.get("content")
    if not isinstance(content, str):
        raise ToolError("REALLMS ASR response did not contain string transcript content")

    # Qwen3-ASR embeds its language label ahead of this delimiter. Other ASR
    # models may return the transcript as plain completion content.
    _, delimiter, transcript = content.partition("<asr_text>")
    text = (transcript if delimiter else content).strip()
    if not text:
        raise ToolError("REALLMS ASR response contained an empty transcript")
    return Transcript(text=text)


def _version() -> str:
    from . import __version__

    return __version__
