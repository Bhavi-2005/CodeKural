import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
from fastapi import HTTPException

from app import sarvam_service
from app.schemas import TestSuggestionOutput as SuggestionOutput


def response(content, finish_reason="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content), finish_reason=finish_reason,
    )])


def suggestion(**overrides):
    values = {
        "purpose": "Protect the selected boundary condition.",
        "test_framework": "pytest",
        "test_code": "def test_edge():\n    assert edge_case() is False",
        "expected_behavior": "The condition returns false for the edge input.",
        "hypothesis_tested": "The guard handles the boundary value.",
        "test_type": "behavioral_regression",
        "limitations": "The repository does not show the project's fixture conventions.",
        "evidence_references": ["E01"],
        "test_objectives": [
            {"test_type": "behavioral_regression", "description": "Check the current boundary behavior.", "testable": True, "evidence_references": ["E01"]},
            {"test_type": "hypothesis_validation", "description": "Check the guard hypothesis.", "testable": True, "evidence_references": ["E01"]},
            {"test_type": "historical_intent_verification", "description": "Requires history, documentation, or author confirmation.", "testable": False, "evidence_references": ["E01"]},
        ],
        "execution_status": "NOT_EXECUTED",
    }
    values.update(overrides)
    return json.dumps(values)


class FakeProviderError(Exception):
    def __init__(self, status):
        self.status_code = status


def install_sdk(monkeypatch, replies, calls):
    class FakeCompletions:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            item = replies.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    class FakeClient:
        def __init__(self, **kwargs):
            calls.append({"client_options": kwargs})
            self.chat = SimpleNamespace(completions=FakeCompletions())

    module = ModuleType("sarvamai")
    module.SarvamAI = FakeClient
    monkeypatch.setitem(sys.modules, "sarvamai", module)
    monkeypatch.setattr(sarvam_service, "SARVAM_API_KEY", "test-key")


def test_strict_schema_and_markdown_fence_are_supported(monkeypatch):
    calls = []
    install_sdk(monkeypatch, [response("```json\n" + suggestion() + "\n```")], calls)

    result = sarvam_service.complete_json("system", "user", SuggestionOutput)

    assert result.evidence_references == ["E01"]
    request = next(item for item in calls if "response_format" in item)
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert request["max_tokens"] == 3000
    assert request["reasoning_effort"] is None


def test_invalid_json_gets_one_bounded_format_retry(monkeypatch):
    calls = []
    install_sdk(monkeypatch, [response("Here is the result: {\"ok\": true}"), response(suggestion())], calls)

    result = sarvam_service.complete_json("system", "user", SuggestionOutput)

    assert result.purpose.startswith("Protect")
    assert len([item for item in calls if "response_format" in item]) == 2


def test_arbitrary_prose_around_json_is_not_extracted(monkeypatch):
    calls = []
    install_sdk(monkeypatch, [response("Answer: " + suggestion())], calls)

    with pytest.raises(HTTPException) as exc:
        sarvam_service.complete_json("system", "user", SuggestionOutput, max_format_retries=0)

    assert exc.value.status_code == 502
    assert len([item for item in calls if "response_format" in item]) == 1


def test_missing_required_test_field_retries_then_validates(monkeypatch):
    calls = []
    incomplete = suggestion().replace('"limitations": "The repository does not show the project\'s fixture conventions.",', "")
    install_sdk(monkeypatch, [response(incomplete), response(suggestion())], calls)

    result = sarvam_service.complete_json("system", "user", SuggestionOutput)

    assert result.limitations
    request_messages = [item["messages"] for item in calls if "response_format" in item]
    assert len(request_messages) == 2
    assert "missing fields" in request_messages[1][-1]["content"].lower()


def test_semantically_invalid_evidence_reference_retries(monkeypatch):
    calls = []
    install_sdk(monkeypatch, [response(suggestion(evidence_references=["E99"])), response(suggestion())], calls)

    result = sarvam_service.complete_json(
        "system", "user", SuggestionOutput,
        semantic_validator=lambda output: (_ for _ in ()).throw(ValueError("unknown evidence ID E99"))
        if output.evidence_references == ["E99"] else None,
    )

    assert result.evidence_references == ["E01"]
    request_messages = [item["messages"] for item in calls if "response_format" in item]
    assert "unknown evidence ID E99" in request_messages[1][-1]["content"]


@pytest.mark.parametrize(("status", "expected"), [(429, 503), (402, 503)])
def test_provider_rate_limit_and_insufficient_credits_are_mapped(monkeypatch, status, expected):
    calls = []
    install_sdk(monkeypatch, [FakeProviderError(status)], calls)

    with pytest.raises(HTTPException) as exc:
        sarvam_service.complete_json("system", "user", SuggestionOutput)

    assert exc.value.status_code == expected
    assert len([item for item in calls if "response_format" in item]) == 1


def test_missing_key_fails_before_sdk_call(monkeypatch):
    monkeypatch.setattr(sarvam_service, "SARVAM_API_KEY", "")
    monkeypatch.setitem(sys.modules, "sarvamai", ModuleType("sarvamai"))

    with pytest.raises(HTTPException) as exc:
        sarvam_service.complete_json("system", "user", SuggestionOutput)

    assert exc.value.status_code == 503
