"""Tests for REALLMS speech recognition."""

from __future__ import annotations

import base64
import json
import unittest
from unittest.mock import patch

from ampav.core.async_tool import ToolError
from ampav.core.schema import ToolOutput, Transcript
from ampav.reallms import ReallmsAsr

from test_text_aboutness import _Response


class ReallmsAsrTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = ReallmsAsr(
            "https://example.test/v1/",
            "test-key",
            model="Qwen3-ASR-1.7B",
            timeout=12,
        )

    @patch("ampav.reallms._chat.urlopen")
    def test_process_posts_audio_data_url(self, urlopen_mock: object) -> None:
        response = {
            "id": "chatcmpl-1",
            "choices": [
                {"message": {"content": "language English<asr_text>A short transcript."}}
            ],
        }
        urlopen_mock.return_value = _Response(json.dumps(response))  # type: ignore[attr-defined]

        result = self.tool.process(
            b"flac bytes",
            media_type="audio/flac",
            temperature=0,
        )

        self.assertIsInstance(result, ToolOutput)
        self.assertEqual(result.tool_name, "reallms_asr")
        self.assertEqual(result.tool_version, "0.0.1")
        self.assertEqual(
            result.parameters,
            {
                "model": "Qwen3-ASR-1.7B",
                "prompt": self.tool.prompt,
                "media_type": "audio/flac",
                "temperature": 0,
            },
        )
        self.assertIsNotNone(result.start_time)
        self.assertIsNotNone(result.end_time)
        self.assertLessEqual(result.start_time, result.end_time)
        self.assertIsInstance(result.output, Transcript)
        self.assertEqual(result.output.text, "A short transcript.")
        self.assertEqual(result.output.words, [])
        self.assertEqual(result.output.paragraphs, [])
        self.assertIsNone(result.output.languages)
        self.assertIsNone(result.tool_private)
        request = urlopen_mock.call_args.args[0]  # type: ignore[attr-defined]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "Qwen3-ASR-1.7B")
        self.assertEqual(payload["temperature"], 0)
        user_content = payload["messages"][0]["content"]
        self.assertEqual(user_content[0]["type"], "text")
        self.assertEqual(user_content[0]["text"], self.tool.prompt)
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

    @patch("ampav.reallms._chat.urlopen")
    def test_process_accepts_plain_transcript_and_can_retain_native_response(
        self, urlopen_mock: object
    ) -> None:
        response = {"choices": [{"message": {"content": "  Plain transcript.  "}}]}
        urlopen_mock.return_value = _Response(json.dumps(response))  # type: ignore[attr-defined]
        tool = ReallmsAsr(
            "https://example.test/v1",
            "test-key",
            model="another-asr-model",
            prompt="Transcribe verbatim.",
            include_tool_private=True,
        )

        result = tool.process(b"audio", media_type="audio/wav")

        self.assertEqual(result.output.text, "Plain transcript.")
        self.assertEqual(result.tool_private, {"native_response": response})

    def test_init_and_process_validate_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "model must not be empty"):
            ReallmsAsr("https://example.test", "test-key", model="")
        with self.assertRaisesRegex(ValueError, "prompt must not be empty"):
            ReallmsAsr("https://example.test", "test-key", model="model", prompt="")
        with self.assertRaisesRegex(ValueError, "audio must not be empty"):
            self.tool.process(b"", media_type="audio/flac")
        with self.assertRaisesRegex(ValueError, "audio MIME type"):
            self.tool.process(b"audio", media_type="video/mp4")
        with self.assertRaisesRegex(ValueError, "configured fields: model, prompt"):
            self.tool.process(
                b"audio",
                media_type="audio/flac",
                model="override",
                prompt="override",
            )

    def test_process_rejects_malformed_native_transcript(self) -> None:
        malformed = [
            ({}, "completion choice"),
            ({"choices": [{}]}, "completion message"),
            ({"choices": [{"message": {"content": ["not text"]}}]}, "string transcript"),
            (
                {"choices": [{"message": {"content": "language English<asr_text>  "}}]},
                "empty transcript",
            ),
        ]
        for response, message in malformed:
            with self.subTest(response=response), patch(
                "ampav.reallms._chat.urlopen",
                return_value=_Response(json.dumps(response)),
            ):
                with self.assertRaisesRegex(ToolError, message):
                    self.tool.process(b"audio", media_type="audio/flac")


if __name__ == "__main__":
    unittest.main()
