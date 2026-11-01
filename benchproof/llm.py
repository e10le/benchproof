"""Calls to OpenAI-compatible model endpoints whose replies must parse into validated objects.

Replies are untrusted: each one goes through a parser that raises ValueError when it is unusable,
and an unusable reply is retried with the problem stated, up to a fixed number of attempts.
"""

import base64
import io
import os
import threading
from collections.abc import Callable

import openai
from PIL import Image

BASETEN_BASE_URL = "https://inference.baseten.co/v1"
META_BASE_URL = "https://api.meta.ai/v1"
SPARK_OPEN_MODEL = "meta-models/muse-spark-open-1026"

# Reasoning is always on and defaults to the slowest, most expensive level; tune on development photos.
REASONING_EFFORT = "medium"
# Reasoning tokens count toward max_tokens, so leave room for the answer after them.
MAX_TOKENS = 8000
REQUEST_TIMEOUT_SECONDS = 120
MAX_IMAGE_BYTES = 10 * 1024 * 1024
IMAGE_MEDIA_TYPES = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}


class ModelError(Exception):
    """A model call did not produce a usable reply."""


class CallBudget:
    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("a call budget needs at least one call")
        self.limit = limit
        self._used = 0
        self._lock = threading.Lock()

    @property
    def used(self) -> int:
        return self._used

    def spend(self) -> None:
        with self._lock:
            if self._used >= self.limit:
                raise ModelError(f"the session used all {self.limit} model calls; start a new session")
            self._used += 1


def baseten_client() -> openai.OpenAI:
    return _client("BASETEN_API_KEY", BASETEN_BASE_URL)


def meta_client() -> openai.OpenAI:
    return _client("MODEL_API_KEY", META_BASE_URL)


def _client(key_variable: str, base_url: str) -> openai.OpenAI:
    api_key = os.environ.get(key_variable)
    if not api_key:
        raise ValueError(f"set {key_variable} in this terminal before calling the model")
    return openai.OpenAI(api_key=api_key, base_url=base_url, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=2)


def image_part(image_bytes: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": data_url(image_bytes)}}


def data_url(image_bytes: bytes) -> str:
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError(f"photo is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB; lower the resolution")

    with Image.open(io.BytesIO(image_bytes)) as image:
        media_type = IMAGE_MEDIA_TYPES.get(image.format)
    if media_type is None:
        raise ValueError(f"photo must be one of {sorted(IMAGE_MEDIA_TYPES)}")

    return f"data:{media_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"


def request_json[T](
    client: openai.OpenAI,
    model: str,
    budget: CallBudget,
    messages: list[dict],
    parse: Callable[[str | None], T],
    max_attempts: int,
) -> T:
    conversation = list(messages)
    problem = ""
    for _ in range(max_attempts):
        budget.spend()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=conversation,
                reasoning_effort=REASONING_EFFORT,
                max_tokens=MAX_TOKENS,
            )
        except openai.APIError as e:
            raise ModelError(f"model request failed: {e}") from e

        content = response.choices[0].message.content
        try:
            return parse(content)
        except ValueError as e:
            problem = str(e)
            conversation += [
                {"role": "assistant", "content": content or ""},
                {"role": "user", "content": f"That reply was not usable: {problem}. Reply with only the JSON object."},
            ]

    raise ModelError(f"model gave no usable reply in {max_attempts} attempts: {problem}")


def json_object(content: str | None) -> str:
    if not content:
        raise ValueError("the reply was empty")

    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end < start:
        raise ValueError("the reply contained no JSON object")
    # Models sometimes wrap JSON in a code fence or a sentence; take the outermost object.
    return content[start : end + 1]
