"""Tests for REALLMS text aboutness generation."""

from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from ampav.core.async_tool import ToolError
from ampav.reallms import ReallmsTextAboutness


class _Response:
    def __init__(self, body: str) -> None:
        self.body = body.encode("utf-8")

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


class ReallmsTextAboutnessTest(unittest.TestCase):
    """Verify aboutness request construction and provider failures."""

    def setUp(self) -> None:
        self.client = ReallmsTextAboutness("https://example.test/v1/", "test-key", timeout=12)

    @patch("ampav.reallms._chat.urlopen")
    def test_process_posts_native_payload_and_returns_object(self, urlopen_mock: object) -> None:
        response = {"id": "chatcmpl-1", "choices": [{"message": {"content": "{}"}}]}
        urlopen_mock.return_value = _Response(json.dumps(response))  # type: ignore[attr-defined]

        result = self.client.process(
            "hello",
            model="glm-5.2",
            temperature=0,
        )

        self.assertEqual(result, response)
        request = urlopen_mock.call_args.args[0]  # type: ignore[attr-defined]
        self.assertEqual(request.full_url, "https://example.test/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(
            json.loads(request.data.decode("utf-8")),
            {
                "model": "glm-5.2",
                "messages": [
                    {"role": "system", "content": "Return JSON only. Do not add markdown fences."},
                    {
                        "role": "user",
                        "content": (
                            "Return a JSON object with summary, named entities, keyphrases, and labels. "
                            "Use labels for broad discovery-oriented concepts useful for finding this transcript."
                            "\n\nTranscript:\nhello"
                        ),
                    },
                ],
                "temperature": 0,
            },
        )
        self.assertEqual(urlopen_mock.call_args.kwargs["timeout"], 12)  # type: ignore[attr-defined]

    @patch("ampav.reallms._chat.urlopen")
    def test_process_wraps_http_errors(self, urlopen_mock: object) -> None:
        urlopen_mock.side_effect = HTTPError("https://example.test", 400, "bad request", {}, io.BytesIO())  # type: ignore[attr-defined]

        with self.assertRaisesRegex(ToolError, "HTTP 400"):
            self.client.process("hello", model="glm-5.2")

    @patch("ampav.reallms._chat.urlopen")
    def test_process_wraps_connection_errors(self, urlopen_mock: object) -> None:
        urlopen_mock.side_effect = URLError("offline")  # type: ignore[attr-defined]

        with self.assertRaisesRegex(ToolError, "could not reach"):
            self.client.process("hello", model="glm-5.2")

    @patch("ampav.reallms._chat.urlopen")
    def test_process_wraps_timeouts(self, urlopen_mock: object) -> None:
        urlopen_mock.side_effect = TimeoutError()  # type: ignore[attr-defined]

        with self.assertRaisesRegex(ToolError, "timed out"):
            self.client.process("hello", model="glm-5.2")

    @patch("ampav.reallms._chat.urlopen")
    def test_process_rejects_non_object_response(self, urlopen_mock: object) -> None:
        urlopen_mock.return_value = _Response("[]")  # type: ignore[attr-defined]

        with self.assertRaisesRegex(ToolError, "non-object"):
            self.client.process("hello", model="glm-5.2")

    def test_process_rejects_empty_text_or_prompt(self) -> None:
        with self.assertRaisesRegex(ValueError, "text must not be empty"):
            self.client.process("", model="glm-5.2")
        with self.assertRaisesRegex(ValueError, "prompt must not be empty"):
            self.client.process("hello", model="glm-5.2", prompt="")



if __name__ == "__main__":
    unittest.main()
