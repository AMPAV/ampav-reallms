"""Thin clients for REALLMS completion endpoints."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ampav.core.async_tool import ToolError


class _ReallmsCompletionClient:
    """Shared native request handling for REALLMS completion endpoints."""

    def __init__(self, base_url: str, api_key: str, *, timeout: float = 60.0) -> None:
        if not base_url:
            raise ValueError("base_url must not be empty")
        if not api_key:
            raise ValueError("api_key must not be empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _process(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Submit a native request and return its decoded native response."""
        request = Request(
            f"{self.base_url}/{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            raise ToolError(f"REALLMS {endpoint} failed with HTTP {error.code}") from error
        except URLError as error:
            raise ToolError(f"REALLMS {endpoint} could not reach the service") from error
        except TimeoutError as error:
            raise ToolError(f"REALLMS {endpoint} timed out") from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ToolError(f"REALLMS {endpoint} returned invalid JSON") from error

        if not isinstance(decoded, dict):
            raise ToolError(f"REALLMS {endpoint} returned a non-object JSON response")
        return decoded


class ReallmsChatCompletions(_ReallmsCompletionClient):
    """Call REALLMS's native synchronous ``/chat/completions`` API.

    ``process`` deliberately returns the decoded native response rather than an
    AMPAV metadata schema. Schema conversion is outside this experiment.
    """

    def process(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **request_options: Any,
    ) -> dict[str, Any]:
        """Submit native chat messages and return the decoded native response."""
        if not model:
            raise ValueError("model must not be empty")
        return self._process(
            "chat/completions",
            {"model": model, "messages": messages, **request_options},
        )


class ReallmsCompletions(_ReallmsCompletionClient):
    """Call REALLMS's native synchronous ``/completions`` API.

    This client is retained only to compare endpoint behavior during the
    experiment; it does not establish a public AMPAV tool contract.
    """

    def process(self, model: str, prompt: str, **request_options: Any) -> dict[str, Any]:
        """Submit a native prompt and return the decoded native response."""
        if not model:
            raise ValueError("model must not be empty")
        if not prompt:
            raise ValueError("prompt must not be empty")
        return self._process(
            "completions",
            {"model": model, "prompt": prompt, **request_options},
        )
