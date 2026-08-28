"""Thin client for REALLMS chat completions."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ampav.core.async_tool import ToolError


class ReallmsChatCompletions:
    """Call REALLMS's native synchronous ``/chat/completions`` API.

    Parameters:
        base_url: REALLMS API base URL, without the endpoint path.
        api_key: API key sent as a bearer token.
        timeout: Default request timeout in seconds.

    ``process`` deliberately returns the decoded native response rather than an
    AMPAV metadata schema.  Schema conversion is outside this experiment.
    """

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

    def process(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **request_options: Any,
    ) -> dict[str, Any]:
        """Submit native chat messages and return the decoded native response.

        Parameters:
            model: REALLMS model identifier.
            messages: Native OpenAI-style chat message objects.
            request_options: Additional native request fields, such as
                ``temperature`` or ``response_format``.

        Raises:
            ToolError: If the service cannot be reached, rejects the request,
                or returns a non-object JSON response.
        """
        if not model:
            raise ValueError("model must not be empty")

        payload = {"model": model, "messages": messages, **request_options}
        request = Request(
            f"{self.base_url}/chat/completions",
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
            raise ToolError(f"REALLMS chat completion failed with HTTP {error.code}") from error
        except URLError as error:
            raise ToolError("REALLMS chat completion could not reach the service") from error
        except TimeoutError as error:
            raise ToolError("REALLMS chat completion timed out") from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ToolError("REALLMS chat completion returned invalid JSON") from error

        if not isinstance(decoded, dict):
            raise ToolError("REALLMS chat completion returned a non-object JSON response")
        return decoded
