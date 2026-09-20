"""Tests for retained multimodal annotation observations."""

from __future__ import annotations

import unittest

from experiments.multimodal_annotation_probe import describe_completion_format


class MultimodalAnnotationProbeTest(unittest.TestCase):
    def test_describes_raw_and_fenced_json_objects(self) -> None:
        self.assertEqual(
            describe_completion_format(
                {"choices": [{"message": {"content": '{"description": "image"}'}}]}
            ),
            "strict JSON object",
        )
        self.assertEqual(
            describe_completion_format(
                {"choices": [{"message": {"content": "```json\n{}\n```"}}]}
            ),
            "fenced JSON object",
        )


if __name__ == "__main__":
    unittest.main()
