"""Text-based aboutness metadata generation through REALLMS."""

from __future__ import annotations

from typing import Any

from ._chat import ReallmsChatClient


DEFAULT_TEXT_ABOUTNESS_PROMPT = (
    "Return a JSON object with summary, named entities, keyphrases, and labels. "
    "Use labels for broad discovery-oriented concepts useful for finding this transcript."
)


class ReallmsTextAboutness:
    """Generate native REALLMS aboutness metadata from text.

    Parameters:
        base_url: REALLMS API base URL, without the endpoint path.
        api_key: API key sent as a bearer token.
        timeout: Default request timeout in seconds.

    ``process`` uses the combined prompt selected by the initial experiment and
    returns the decoded native response. AMPAV schema conversion remains a
    separate integration decision.
    """

    def __init__(self, base_url: str, api_key: str, *, timeout: float = 60.0) -> None:
        self._client = ReallmsChatClient(base_url, api_key, timeout=timeout)

    def process(
        self,
        text: str,
        *,
        model: str,
        prompt: str = DEFAULT_TEXT_ABOUTNESS_PROMPT,
        **request_options: Any,
    ) -> dict[str, Any]:
        """Generate aboutness metadata for ``text`` and return native output.

        ``prompt`` is the metadata instruction placed before the transcript. It
        is configurable so prompt experiments do not require transport changes.
        Additional keyword arguments are passed through as native request fields.
        """
        if not text:
            raise ValueError("text must not be empty")
        if not prompt:
            raise ValueError("prompt must not be empty")

        return self._client.complete(
            model,
            [
                {"role": "system", "content": "Return JSON only. Do not add markdown fences."},
                {"role": "user", "content": f"{prompt}\n\nTranscript:\n{text}"},
            ],
            **request_options,
        )
