"""Tests for REALLMS multimodal annotation generation."""

from __future__ import annotations

import base64
import json
import unittest
from unittest.mock import patch

from ampav.reallms import ReallmsMultimodalAnnotation

from test_text_aboutness import _Response


class ReallmsMultimodalAnnotationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = ReallmsMultimodalAnnotation(
            "https://example.test/v1/", "test-key", timeout=12
        )

    @patch("ampav.reallms._chat.urlopen")
    def test_process_posts_image_data_url(self, urlopen_mock: object) -> None:
        response = {"id": "chatcmpl-1", "choices": [{"message": {"content": "{}"}}]}
        urlopen_mock.return_value = _Response(json.dumps(response))  # type: ignore[attr-defined]

        result = self.tool.process(
            b"jpeg bytes",
            media_type="image/jpeg",
            model="gemma-4-31B-it",
            temperature=0,
        )

        self.assertEqual(result, response)
        request = urlopen_mock.call_args.args[0]  # type: ignore[attr-defined]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "gemma-4-31B-it")
        self.assertEqual(payload["temperature"], 0)
        user_content = payload["messages"][1]["content"]
        self.assertEqual(user_content[0]["type"], "text")
        self.assertEqual(
            user_content[1],
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        "data:image/jpeg;base64,"
                        + base64.b64encode(b"jpeg bytes").decode("ascii")
                    )
                },
            },
        )

    def test_process_validates_image_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "image must not be empty"):
            self.tool.process(b"", media_type="image/jpeg", model="gemma-4-31B-it")
        with self.assertRaisesRegex(ValueError, "image MIME type"):
            self.tool.process(b"image", media_type="audio/flac", model="gemma-4-31B-it")
        with self.assertRaisesRegex(ValueError, "prompt must not be empty"):
            self.tool.process(
                b"image", media_type="image/jpeg", model="gemma-4-31B-it", prompt=""
            )


if __name__ == "__main__":
    unittest.main()
