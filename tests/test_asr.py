"""Tests for REALLMS speech recognition."""

from __future__ import annotations

import base64
import json
import unittest
from unittest.mock import patch

from ampav.reallms import ReallmsAsr

from test_text_aboutness import _Response


class ReallmsAsrTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = ReallmsAsr("https://example.test/v1/", "test-key", timeout=12)

    @patch("ampav.reallms._chat.urlopen")
    def test_process_posts_audio_data_url(self, urlopen_mock: object) -> None:
        response = {
            "id": "chatcmpl-1",
            "choices": [{"message": {"content": "A short transcript."}}],
        }
        urlopen_mock.return_value = _Response(json.dumps(response))  # type: ignore[attr-defined]

        result = self.tool.process(
            b"flac bytes",
            media_type="audio/flac",
            model="Qwen3-ASR-1.7B",
            temperature=0,
        )

        self.assertEqual(result, response)
        request = urlopen_mock.call_args.args[0]  # type: ignore[attr-defined]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "Qwen3-ASR-1.7B")
        self.assertEqual(payload["temperature"], 0)
        user_content = payload["messages"][0]["content"]
        self.assertEqual(user_content[0]["type"], "text")
        self.assertEqual(
            user_content[1],
            {
                "type": "audio_url",
                "audio_url": {
                    "url": (
                        "data:audio/flac;base64,"
                        + base64.b64encode(b"flac bytes").decode("ascii")
                    )
                },
            },
        )

    def test_process_validates_audio_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "audio must not be empty"):
            self.tool.process(b"", media_type="audio/flac", model="Qwen3-ASR-1.7B")
        with self.assertRaisesRegex(ValueError, "audio MIME type"):
            self.tool.process(b"audio", media_type="video/mp4", model="Qwen3-ASR-1.7B")
        with self.assertRaisesRegex(ValueError, "prompt must not be empty"):
            self.tool.process(
                b"audio", media_type="audio/flac", model="Qwen3-ASR-1.7B", prompt=""
            )


if __name__ == "__main__":
    unittest.main()
