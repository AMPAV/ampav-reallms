"""Multimodal annotation generation through REALLMS."""

from __future__ import annotations

import base64
from typing import Any

from ._chat import ReallmsChatClient


DEFAULT_MULTIMODAL_ANNOTATION_PROMPT = (
    "Return a JSON object describing the image, including visible text, named "
    "entities, people, objects, activities or events, setting, and overall subjects. "
    "Distinguish directly observed details from inferences and identify uncertainty."
)


class ReallmsMultimodalAnnotation:
    """Generate native REALLMS annotations from supported media.

    The current boundary accepts one image because image input is the verified
    REALLMS capability. Additional modalities should be added only after their
    native message representation has been demonstrated.
    """

    def __init__(self, base_url: str, api_key: str, *, timeout: float = 60.0) -> None:
        self._client = ReallmsChatClient(base_url, api_key, timeout=timeout)

    def process(
        self,
        image: bytes,
        *,
        media_type: str,
        model: str,
        prompt: str = DEFAULT_MULTIMODAL_ANNOTATION_PROMPT,
        **request_options: Any,
    ) -> dict[str, Any]:
        """Annotate one image and return the decoded native response.

        ``media_type`` is the image MIME type used in the inline data URL.
        ``prompt`` is configurable so annotation instructions can evolve
        independently of the REALLMS transport.
        """
        if not image:
            raise ValueError("image must not be empty")
        if not media_type.startswith("image/"):
            raise ValueError("media_type must be an image MIME type")
        if not prompt:
            raise ValueError("prompt must not be empty")

        encoded_image = base64.b64encode(image).decode("ascii")
        return self._client.complete(
            model,
            [
                {
                    "role": "system",
                    "content": "Return JSON only. Do not add markdown fences.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{encoded_image}",
                            },
                        },
                    ],
                },
            ],
            **request_options,
        )
