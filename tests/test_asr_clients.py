"""Tests for REALLMS ASR pipeline and CLI clients."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import ANY, patch

from ampav.core.schema import ToolOutput, Transcript
from ampav.reallms.asr import DEFAULT_ASR_PROMPT
from ampav_reallms_cli.asr import build_cli_parser, main as cli_main
from ampav_reallms_pipeline import transcribe_file


OPEN_DOOR = Path(__file__).parents[1] / "examples" / "data" / "OpenDoor.mp3"


def tool_output() -> ToolOutput:
    return ToolOutput(
        tool_name="reallms_asr",
        tool_version="test",
        output=Transcript(text="Please open the door."),
    )


class ReallmsAsrClientsTest(unittest.TestCase):
    @patch("ampav_reallms_pipeline.asr.ReallmsAsr")
    def test_transcribe_file_reads_local_audio_and_dispatches_tool(
        self, tool_class: object
    ) -> None:
        expected = tool_output()
        tool_class.return_value.process.return_value = expected  # type: ignore[attr-defined]

        result = transcribe_file(
            OPEN_DOOR,
            base_url="https://example.test/v1",
            api_key="test-key",
            model="Qwen3-ASR-1.7B",
            temperature=0,
        )

        self.assertIs(result, expected)
        tool_class.assert_called_once_with(  # type: ignore[attr-defined]
            "https://example.test/v1",
            "test-key",
            model="Qwen3-ASR-1.7B",
            prompt=DEFAULT_ASR_PROMPT,
            timeout=60.0,
            include_tool_private=False,
        )
        tool_class.return_value.process.assert_called_once_with(  # type: ignore[attr-defined]
            OPEN_DOOR.resolve().read_bytes(),
            media_type="audio/mpeg",
            temperature=0,
        )

    def test_transcribe_file_validates_path_and_media_type(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "Input file does not exist"):
            transcribe_file(
                "missing-audio.mp3",
                base_url="https://example.test/v1",
                api_key="test-key",
                model="model",
            )

        with TemporaryDirectory() as directory:
            unknown = Path(directory) / "audio.unknown"
            unknown.write_bytes(b"audio")
            with self.assertRaisesRegex(ValueError, "pass media_type"):
                transcribe_file(
                    unknown,
                    base_url="https://example.test/v1",
                    api_key="test-key",
                    model="model",
                )

    def test_cli_parser_and_pipeline_dispatch(self) -> None:
        args = build_cli_parser().parse_args(
            [
                "speech.mp3",
                "--base-url",
                "https://example.test/v1",
                "--model",
                "test-asr",
                "--media-type",
                "audio/mpeg",
                "--include-tool-private",
            ]
        )
        self.assertEqual(args.media, Path("speech.mp3"))
        self.assertEqual(args.model, "test-asr")
        self.assertEqual(args.media_type, "audio/mpeg")
        self.assertTrue(args.include_tool_private)

        expected = tool_output()
        with (
            patch.dict(os.environ, {"REALLMS_API_KEY": "test-key"}),
            patch("ampav_reallms_cli.asr.transcribe_file", return_value=expected) as transcribe,
            patch("builtins.print") as print_output,
        ):
            exit_code = cli_main(
                [
                    "speech.mp3",
                    "--base-url",
                    "https://example.test/v1",
                    "--model",
                    "test-asr",
                ]
            )

        self.assertEqual(exit_code, 0)
        transcribe.assert_called_once_with(
            Path("speech.mp3"),
            base_url="https://example.test/v1",
            api_key="test-key",
            model="test-asr",
            prompt=DEFAULT_ASR_PROMPT,
            media_type=None,
            include_tool_private=False,
            timeout=60.0,
            temperature=0.0,
        )
        print_output.assert_called_once_with(expected.model_dump_yaml(sort_keys=False))

    def test_cli_reports_missing_api_key(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("ampav_reallms_cli.asr.logging.error") as log_error,
        ):
            exit_code = cli_main(
                ["speech.mp3", "--base-url", "https://example.test/v1"]
            )

        self.assertEqual(exit_code, 1)
        log_error.assert_called_once_with("%s", ANY)


if __name__ == "__main__":
    unittest.main()
