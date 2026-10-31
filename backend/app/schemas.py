from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


EVIDENCE_KIND_BY_TYPE = {
    "source": "SOURCE_FACT",
    "source_function": "SOURCE_FACT",
    "related_function": "SOURCE_FACT",
    "assignment": "SOURCE_FACT",
    "read": "SOURCE_FACT",
    "documentation": "DOCUMENTATION_FACT",
    "commit": "GIT_HISTORY",
    "blame": "GIT_HISTORY",
    "line_history": "GIT_HISTORY",
    "diff": "GIT_HISTORY",
    "test": "TEST_EVIDENCE",
    "test_configuration": "CONFIGURATION_EVIDENCE",
    "dependency_configuration": "CONFIGURATION_EVIDENCE",
}

EVIDENCE_CATEGORY_BY_TYPE = {
    "source": "A_source",
    "source_function": "A_source",
    "related_function": "D_cross_references",
    "assignment": "D_cross_references",
    "read": "D_cross_references",
    "documentation": "F_documentation",
    "commit": "B_history",
    "blame": "B_history",
    "line_history": "C_diff",
    "diff": "C_diff",
    "test": "E_tests",
    "test_configuration": "G_configuration",
    "dependency_configuration": "G_configuration",
}


class ValidateRepositoryRequest(BaseModel):
    path: str = Field(min_length=1, max_length=2048)


class InvestigationRequest(BaseModel):
    repository_id: str
    file_path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    question: str = Field(min_length=3, max_length=1000)
    language: Literal["en", "ta"] = "en"
    dependency_depth: int = Field(default=2, ge=0, le=5)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class EvidenceRecord(StrictModel):
    id: str = Field(pattern=r"^E\d{2}$")
    type: str = Field(min_length=2, max_length=40)
    evidence_kind: Literal["SOURCE_FACT", "DOCUMENTATION_FACT", "GIT_HISTORY", "TEST_EVIDENCE", "CONFIGURATION_EVIDENCE"] | None = None
    category: str = Field(default="A_source", min_length=1, max_length=40)
    location: str = Field(min_length=1, max_length=500)
    path: str | None = Field(default=None, max_length=1000)
    commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{40,64}$")
    line_range: str | None = Field(default=None, max_length=40)
    url: str | None = Field(default=None, max_length=1000)
    excerpt: str = Field(min_length=1, max_length=16_000)
    relevance_reason: str = Field(min_length=1, max_length=500)
    relevance_rank: int = Field(default=99, ge=0, le=99)


class EvidenceClaim(StrictModel):
    text: str = Field(min_length=3, max_length=1200)
    evidence_references: list[str] = Field(min_length=1, max_length=12)
    hypothesis_type: Literal["SOURCE_FACT", "BEHAVIORAL_HYPOTHESIS", "TECHNICAL_HYPOTHESIS", "DOCUMENTED_ARCHITECTURAL_ROLE", "ARCHITECTURAL_HYPOTHESIS", "HISTORICAL_HYPOTHESIS"] | None = None


class HypothesisRecord(StrictModel):
    hypothesis_id: str = Field(pattern=r"^H[0-9A-F]{8}$")
    hypothesis_text: str = Field(min_length=3, max_length=1200)
    hypothesis_type: Literal["BEHAVIORAL_HYPOTHESIS", "TECHNICAL_HYPOTHESIS", "DOCUMENTED_ARCHITECTURAL_ROLE", "ARCHITECTURAL_HYPOTHESIS", "HISTORICAL_HYPOTHESIS"]
    testability: bool
    evidence_references: list[str] = Field(min_length=1, max_length=12)
    validation_method: str = Field(min_length=3, max_length=1000)
    limitations: str = Field(min_length=3, max_length=1000)


class TestSuggestionRequest(BaseModel):
    investigation: InvestigationRequest
    framework: str | None = Field(default=None, max_length=80)
    hypothesis: HypothesisRecord


class EvidenceContradiction(StrictModel):
    description: str = Field(min_length=3, max_length=1200)
    why_it_matters: str = Field(min_length=3, max_length=800)
    evidence_references: list[str] = Field(min_length=2, max_length=12)


class InvestigationOutput(StrictModel):
    summary: str = Field(min_length=3, max_length=1600)
    summary_evidence_references: list[str] = Field(min_length=1, max_length=12)
    source_observations: list[EvidenceClaim] = Field(max_length=20)
    historical_facts: list[EvidenceClaim] = Field(max_length=20)
    hypotheses: list[EvidenceClaim] = Field(max_length=20)
    unknowns: list[str] = Field(max_length=20)
    contradictions: list[EvidenceContradiction] = Field(max_length=10)


class TestObjective(StrictModel):
    test_type: Literal["behavioral_regression", "hypothesis_validation", "historical_intent_verification"]
    description: str = Field(min_length=3, max_length=500)
    testable: bool
    evidence_references: list[str] = Field(min_length=1, max_length=12)


class TestSuggestionOutput(StrictModel):
    purpose: str = Field(min_length=3, max_length=800)
    test_framework: str = Field(min_length=2, max_length=120)
    test_code: str = Field(min_length=3, max_length=6000)
    expected_behavior: str = Field(min_length=3, max_length=800)
    hypothesis_tested: str = Field(min_length=3, max_length=800)
    test_type: Literal["behavioral_regression", "hypothesis_validation"]
    limitations: str = Field(min_length=3, max_length=1000)
    evidence_references: list[str] = Field(min_length=1, max_length=12)
    test_objectives: list[TestObjective] = Field(min_length=3, max_length=3)
    execution_status: Literal["NOT_EXECUTED"]
    hypothesis_id: str | None = Field(default=None, pattern=r"^H[0-9A-F]{8}$")
    hypothesis_type: Literal["BEHAVIORAL_HYPOTHESIS", "TECHNICAL_HYPOTHESIS"] | None = None
