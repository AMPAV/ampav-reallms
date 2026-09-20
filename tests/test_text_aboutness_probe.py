"""Tests for safe retained REALLMS experiment output."""

from __future__ import annotations

import unittest

from experiments.text_aboutness_probe import (
    describe_completion_format,
    has_provider_reasoning,
    sanitize_response,
)


class SanitizeResponseTest(unittest.TestCase):
    """Verify provider reasoning is excluded without reshaping native output."""

    def test_removes_reasoning_from_message_fields(self) -> None:
        response = {
            "id": "completion-1",
            "choices": [
                {
                    "message": {
                        "content": "{}",
                        "reasoning_content": "private chain",
                        "provider_specific_fields": {
                            "reasoning": "duplicate private chain",
                            "refusal": None,
                        },
                    }
                }
            ],
        }

        sanitized = sanitize_response(response)

        self.assertEqual(response["choices"][0]["message"]["reasoning_content"], "private chain")
        self.assertEqual(sanitized["id"], "completion-1")
        message = sanitized["choices"][0]["message"]
        self.assertNotIn("reasoning_content", message)
        self.assertEqual(message["provider_specific_fields"], {"refusal": None})

    def test_describes_strict_json_and_provider_reasoning(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": '{"summary": "text"}',
                        "reasoning_content": "private chain",
                    }
                }
            ]
        }

        self.assertEqual(describe_completion_format(response), "strict JSON object")
        self.assertTrue(has_provider_reasoning(response))
        self.assertEqual(
            describe_completion_format({"choices": [{"message": {"content": "```json\\n{}\\n```"}}]}),
            "not strict JSON",
        )


if __name__ == "__main__":
    unittest.main()
