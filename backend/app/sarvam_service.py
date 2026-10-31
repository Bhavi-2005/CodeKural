import json
import logging
import re
from collections.abc import Callable
from typing import TypeVar

from fastapi import HTTPException
from pydantic import BaseModel, ValidationError

from .config import SARVAM_API_KEY, SARVAM_MODEL

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
_FENCED_JSON = re.compile(r"^\s*```(?:json)?\s*\r?\n?(.*?)\r?\n?```\s*$", re.IGNORECASE | re.DOTALL)


def complete_json(system_prompt: str, user_prompt: str, output_model: type[T], *,
                  semantic_validator: Callable[[T], None] | None = None,
                  max_format_retries: int = 1) -> T:
    """Request a schema-constrained Sarvam response and validate it locally.

    Only JSON parsing/schema failures get one corrective generation. Provider and
    transport failures are surfaced immediately and never cause duplicate calls.
    """
    if not SARVAM_API_KEY:
        raise HTTPException(503, "Sarvam is not configured. Set SARVAM_API_KEY in the backend .env file to enable AI actions.")

    schema = output_model.model_json_schema()
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": output_model.__name__,
            "description": f"Strict response structure for {output_model.__name__}.",
            "strict": True,
            "schema": schema,
        },
    }
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    attempts = max(1, min(max_format_retries + 1, 2))

    for attempt in range(attempts):
        response = _call_sarvam(messages, response_format)
        content, finish_reason = _read_content(response)
        try:
            if finish_reason == "length" or not content:
                raise ValueError("incomplete response")
            parsed = _parse_json_object(content)
            validated = output_model.model_validate(parsed)
            if semantic_validator:
                semantic_validator(validated)
            return validated
        except (ValueError, json.JSONDecodeError, ValidationError) as exc:
            if attempt + 1 >= attempts:
                logger.warning("Sarvam structured response failed validation for %s", output_model.__name__)
                raise HTTPException(502, "Sarvam returned a response that did not match the required structure. Retry the request.")
            hint = _format_hint(exc)
            messages[1]["content"] = (
                user_prompt + "\n\nThe previous generation failed local JSON/schema validation. "
                "Return a corrected object matching the same strict schema. " + hint +
                " Do not add prose or Markdown fences."
            )
    raise HTTPException(502, "Sarvam returned a response that did not match the required structure. Retry the request.")


def _call_sarvam(messages: list[dict[str, str]], response_format: dict) -> object:
    try:
        from sarvamai import SarvamAI

        client = SarvamAI(api_subscription_key=SARVAM_API_KEY, timeout=45.0)
        return client.chat.completions(
            model=SARVAM_MODEL,
            messages=messages,
            max_tokens=3000,
            reasoning_effort=None,
            response_format=response_format,
            request_options={"max_retries": 1},
        )
    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(503, "Sarvam SDK is missing. Install backend requirements and restart the API.")
    except Exception as exc:
        status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        response = getattr(exc, "response", None)
        status = status or getattr(response, "status_code", None)
        logger.warning("Sarvam request failed (%s)", type(exc).__name__)
        if status in (401, 403):
            raise HTTPException(502, "Sarvam rejected the API key. Check SARVAM_API_KEY.")
        if status == 402:
            raise HTTPException(503, "Sarvam account credits are insufficient. Check your account billing.")
        if status == 429:
            raise HTTPException(503, "Sarvam rate limit reached. Try again later.")
        if status in (400, 404, 422):
            raise HTTPException(502, "Sarvam rejected the model request. Check SARVAM_MODEL and update the SDK.")
        if "timeout" in type(exc).__name__.lower() or status == 408:
            raise HTTPException(504, "Sarvam took too long to respond. Please retry.")
        raise HTTPException(502, "Sarvam is temporarily unavailable. Please try again shortly.")


def _read_content(response: object) -> tuple[str | None, str | None]:
    try:
        choice = response.choices[0]  # Official SDK returns typed response models with attribute access.
        message = choice.message
        content = message.content
        finish_reason = choice.finish_reason
    except (AttributeError, IndexError, TypeError):
        return None, None
    return (content if isinstance(content, str) else None,
            finish_reason if isinstance(finish_reason, str) else None)


def _parse_json_object(content: str) -> dict:
    candidate = content.strip()
    fenced = _FENCED_JSON.fullmatch(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("top-level JSON must be an object")
    return parsed


def _format_hint(error: Exception) -> str:
    if isinstance(error, ValidationError):
        fields = [".".join(str(part) for part in item["loc"]) for item in error.errors()[:8]]
        return "Invalid or missing fields: " + (", ".join(fields) or "schema mismatch") + "."
    if isinstance(error, json.JSONDecodeError):
        return "The previous response was not valid JSON."
    if isinstance(error, ValueError) and str(error):
        return str(error)[:300]
    return "The previous response was empty, truncated, or not a JSON object."
