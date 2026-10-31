import ast
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.evidence_service import _coverage_measurement, _rank_evidence, collect_evidence
from app.config import SARVAM_API_KEY
from app.git_service import file_history
from app.investigation_service import (_build_candidate_score_suggestion, _candidate_score_contract,
                                      _deduplicate_hypotheses, _derive_candidate_score_hypotheses, _evidence_reason_matches_type,
                                      _build_preprocessing_suggestion, _derive_preprocessing_hypotheses,
                                      _derive_preprocessing_source_observations,
                                      _hypothesis_has_support, _normalize_test_references, _validate_evidence, _validate_test_code,
                                      _validate_test_suggestion, _build_extract_plate_suggestion, _classify_hypothesis,
                                      _extract_plate_test_code, _separate_source_facts, _validate_claim_support,
                                      _validate_investigation_consistency, _validate_investigation_references,
                                      _validate_references, _python_technical_cues, _mentions_historical_motivation, investigate)
from app.repository_service import RepositoryStore
from app.schemas import EvidenceClaim, HypothesisRecord, InvestigationOutput, InvestigationRequest, TestSuggestionOutput as SuggestionOutput


PLATE_EXTRACTOR_TEST_CODE = _extract_plate_test_code()


PREPROCESSING_SOURCE = '''"""Image transformations used by the detection and OCR stages."""
from __future__ import annotations
import cv2
import numpy as np

def validate_image(image: np.ndarray) -> None:
    """Raise a ValueError when an image cannot be safely processed."""
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a NumPy array.")
    if image.size == 0:
        raise ValueError("Image is empty.")
    if image.ndim not in (2, 3):
        raise ValueError("Image must have two or three dimensions.")
    if image.ndim == 3 and image.shape[2] not in (3, 4):
        raise ValueError("Color images must have three or four channels.")

def resize_for_processing(image: np.ndarray, max_width: int = 1000) -> tuple[np.ndarray, float]:
    """Downscale a large image and return it with the applied scale factor."""
    validate_image(image)
    height, width = image.shape[:2]
    if width <= max_width:
        return image.copy(), 1.0
    scale = max_width / width
    resized = cv2.resize(image, (max_width, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    return resized, scale

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert grayscale, BGR, or BGRA images to a single channel array."""
    validate_image(image)
    if image.ndim == 2:
        return image.copy()
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def prepare_for_detection(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Produce a denoised grayscale image and Canny edge map."""
    gray = to_grayscale(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 60, 180)
    return gray, edges

def enhance_plate_for_ocr(plate_image: np.ndarray) -> np.ndarray:
    """Upscale, denoise, and binarize a plate crop for OCR."""
    validate_image(plate_image)
    gray = to_grayscale(plate_image)
    enlarged = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.bilateralFilter(enlarged, 7, 50, 50)
    thresholded = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 31, 9)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel, iterations=1)
'''


PREPROCESSING_TESTS_SOURCE = '''import numpy as np
import pytest
from src.plate_detector import detect_license_plate
from src.preprocessing import enhance_plate_for_ocr, prepare_for_detection, validate_image

def test_prepare_for_detection_returns_grayscale_and_edges():
    image = np.zeros((80, 160, 3), dtype=np.uint8)
    image[25:55, 40:120] = 255
    gray, edges = prepare_for_detection(image)
    assert gray.shape == (80, 160)
    assert edges.shape == (80, 160)
    assert edges.dtype == np.uint8
    assert np.count_nonzero(edges) > 0

def test_enhance_plate_for_ocr_returns_larger_binary_image():
    plate = np.full((20, 80, 3), 220, dtype=np.uint8)
    enhanced = enhance_plate_for_ocr(plate)
    assert enhanced.shape == (60, 240)
    assert set(np.unique(enhanced)).issubset({0, 255})

def test_invalid_image_input_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        validate_image(np.array([], dtype=np.uint8))

def test_blank_image_has_no_plate_candidate():
    blank_image = np.zeros((300, 500, 3), dtype=np.uint8)
    assert detect_license_plate(blank_image) is None
'''


def _collect_preprocessing_evidence(tmp_path, monkeypatch, selected_function: str | None = "to_grayscale"):
    source = PREPROCESSING_SOURCE
    repo = make_repo(tmp_path / f"preprocessing_{selected_function or 'all'}", {
        "src/__init__.py": "",
        "src/preprocessing.py": source,
        "src/plate_detector.py": '''from src.preprocessing import prepare_for_detection, resize_for_processing, validate_image

def detect_license_plate(image):
    validate_image(image)
    resized, _ = resize_for_processing(image)
    prepare_for_detection(resized)
    return None
''',
        "tests/__init__.py": "",
        "tests/test_preprocessing.py": PREPROCESSING_TESTS_SOURCE,
        "requirements.txt": "numpy>=1.26\npytest>=8.0\nopencv-python-headless>=4.10\n",
        "README.md": "The preprocessing module validates and prepares image data for plate detection and OCR.\n",
    })
    store, repo_id = repository_store(monkeypatch, tmp_path, repo)
    lines = source.splitlines()
    if selected_function is None:
        start, end = 1, len(lines)
    else:
        start = next(index for index, line in enumerate(lines, 1) if line.startswith(f"def {selected_function}("))
        end = next((index for index in range(start + 1, len(lines) + 1) if lines[index - 1].startswith("def ")), len(lines) + 1) - 1
    _, evidence, _, scope = collect_evidence(store, repo_id, "src/preprocessing.py", start, end,
                                             f"Investigate {selected_function or 'preprocessing'}", dependency_depth=3)
    return store, repo_id, evidence, scope, repo


def test_preprocessing_hypotheses_match_source_contracts_and_have_evidence(tmp_path, monkeypatch):
    _, _, evidence, _, _ = _collect_preprocessing_evidence(tmp_path, monkeypatch, None)
    claims = _derive_preprocessing_hypotheses(evidence)
    assert len(claims) == 7
    ids = {item["id"]: item for item in evidence}
    assert all(claim.hypothesis_type == "BEHAVIORAL_HYPOTHESIS" for claim in claims)
    assert all(claim.evidence_references and all(reference in ids for reference in claim.evidence_references)
               for claim in claims)
    assert any("max_width" in claim.text and "scale" in claim.text for claim in claims)
    assert any("BGR uint8" in claim.text for claim in claims)
    assert any("BGRA uint8" in claim.text for claim in claims)
    assert any("two-channel" in claim.text for claim in claims)
    assert any("independent copy" in claim.text for claim in claims)

    observations = _derive_preprocessing_source_observations(evidence)
    for function_name in ("validate_image", "resize_for_processing", "to_grayscale",
                          "prepare_for_detection", "enhance_plate_for_ocr"):
        assert sum(function_name in claim.text for claim in observations) == 1


def test_preprocessing_hypotheses_are_suppressed_without_matching_complete_source():
    partial = [{"id": "E01", "type": "source_function", "path": "src/preprocessing.py",
                "excerpt": "1: def to_grayscale(image):\n2:     return image.copy()"}]
    assert _derive_preprocessing_hypotheses(partial) == []


def test_preprocessing_test_discovery_uses_direct_and_indirect_call_chains(tmp_path, monkeypatch):
    expectations = {
        "validate_image": {
            "test_invalid_image_input_is_rejected": "DIRECT",
            "test_blank_image_has_no_plate_candidate": "INDIRECT",
            "test_prepare_for_detection_returns_grayscale_and_edges": "INDIRECT",
            "test_enhance_plate_for_ocr_returns_larger_binary_image": "INDIRECT",
        },
        "resize_for_processing": {"test_blank_image_has_no_plate_candidate": "INDIRECT"},
        "to_grayscale": {
            "test_blank_image_has_no_plate_candidate": "INDIRECT",
            "test_prepare_for_detection_returns_grayscale_and_edges": "INDIRECT",
            "test_enhance_plate_for_ocr_returns_larger_binary_image": "INDIRECT",
        },
        "prepare_for_detection": {
            "test_prepare_for_detection_returns_grayscale_and_edges": "DIRECT",
            "test_blank_image_has_no_plate_candidate": "INDIRECT",
        },
        "enhance_plate_for_ocr": {
            "test_enhance_plate_for_ocr_returns_larger_binary_image": "DIRECT",
        },
    }
    for selected, expected in expectations.items():
        _, _, _, scope, _ = _collect_preprocessing_evidence(tmp_path, monkeypatch, selected)
        refs = {name: item["classification"] for item in scope["test_discovery"]["references"]
                for name in item["test_names"]}
        for test_name, classification in expected.items():
            assert refs[test_name] == classification


def test_preprocessing_observations_are_individual_and_documentation_is_not_historical(tmp_path, monkeypatch):
    _, _, evidence, _, _ = _collect_preprocessing_evidence(tmp_path, monkeypatch, None)
    observations = _derive_preprocessing_source_observations(evidence)
    claims = _derive_preprocessing_hypotheses(evidence)
    assert len(observations) == 5
    assert all(claim.hypothesis_type == "SOURCE_FACT" for claim in observations)
    assert all(claim.hypothesis_type == "BEHAVIORAL_HYPOTHESIS" for claim in claims)
    README = next(item for item in evidence if item["type"] == "documentation")
    assert README["evidence_kind"] == "DOCUMENTATION_FACT"
    assert all(README["id"] not in claim.evidence_references for claim in claims)
    assert "historical intent" in _classify_hypothesis(
        EvidenceClaim(text="The preprocessing module owns input validation.",
                      evidence_references=[README["id"]], hypothesis_type="DOCUMENTED_ARCHITECTURAL_ROLE"),
        {item["id"]: item for item in evidence})["limitations"]


def test_investigation_response_includes_separate_preprocessing_observations(tmp_path, monkeypatch):
    store, repo_id, evidence, _, _ = _collect_preprocessing_evidence(tmp_path, monkeypatch, None)
    source_id = next(item["id"] for item in evidence if item["type"] == "source")
    request = InvestigationRequest(repository_id=repo_id, file_path="src/preprocessing.py", start_line=1,
                                   end_line=len(PREPROCESSING_SOURCE.splitlines()),
                                   question="Describe the preprocessing functions separately.", language="en")

    def fake_complete(system, prompt, output_model, semantic_validator=None, **kwargs):
        result = output_model.model_validate({
            "summary": "The module contains distinct image validation and transformation functions.",
            "summary_evidence_references": [source_id], "source_observations": [], "historical_facts": [],
            "hypotheses": [], "unknowns": [], "contradictions": [],
        })
        if semantic_validator:
            semantic_validator(result)
        return result

    monkeypatch.setattr("app.investigation_service.complete_json", fake_complete)
    response = investigate(store, request)
    observations = response["source_observations"]
    for function_name in ("validate_image", "resize_for_processing", "to_grayscale",
                          "prepare_for_detection", "enhance_plate_for_ocr"):
        matches = [item for item in observations if function_name in item["text"]]
        assert len(matches) == 1
        assert matches[0]["classification"] == "SOURCE_FACT"
        assert matches[0]["evidence_references"]
        assert all(reference in {item["id"] for item in response["evidence"]}
                   for reference in matches[0]["evidence_references"])
    assert len(response["hypotheses"]) == 7
    assert all(item["hypothesis_type"] == "BEHAVIORAL_HYPOTHESIS" for item in response["hypotheses"])
    evidence_ids = {item["id"] for item in response["evidence"]}
    for item in response["hypotheses"]:
        proposal = item["test_proposal"]
        assert proposal["hypothesis_id"] == item["hypothesis_id"]
        assert proposal["hypothesis_tested"] == item["hypothesis_text"]
        assert proposal["execution_status"] == "NOT_EXECUTED"
        assert ast.parse(proposal["test_code"])
        assert set(proposal["evidence_references"]) <= evidence_ids


def test_preprocessing_proposals_validate_and_remain_not_executed(tmp_path, monkeypatch):
    store, repo_id, evidence, _, _ = _collect_preprocessing_evidence(tmp_path, monkeypatch, None)
    index = 0
    for claim in _derive_preprocessing_hypotheses(evidence):
        index += 1
        hypothesis = HypothesisRecord(hypothesis_id=f"H{index:08X}", hypothesis_text=claim.text,
                                      hypothesis_type=claim.hypothesis_type, testability=True,
                                      evidence_references=claim.evidence_references,
                                      validation_method="Check deterministic NumPy inputs against the inspected implementation.",
                                      limitations="Does not establish original developer intent.")
        suggestion = _build_preprocessing_suggestion(store, repo_id, "src/preprocessing.py", evidence, hypothesis)
        assert suggestion is not None, claim.text
        tree = ast.parse(suggestion.test_code)
        assert any(node.name.startswith("test_") for node in tree.body if isinstance(node, ast.FunctionDef))
        assert suggestion.execution_status == "NOT_EXECUTED"
        assert all(reference in {item["id"] for item in evidence} for reference in suggestion.evidence_references)
        assert "original developer intent" in suggestion.limitations
    assert index == 7


def test_investigate_routes_preprocessing_hypothesis_to_validated_proposal(tmp_path, monkeypatch):
    store, repo_id, evidence, _, _ = _collect_preprocessing_evidence(tmp_path, monkeypatch, "resize_for_processing")
    claim = next(item for item in _derive_preprocessing_hypotheses(evidence)
                 if "wider than max_width" in item.text)
    hypothesis = HypothesisRecord(hypothesis_id="HABCDEF12", hypothesis_text=claim.text,
                                  hypothesis_type=claim.hypothesis_type, testability=True,
                                  evidence_references=claim.evidence_references,
                                  validation_method="Check dimensions and scale for a deterministic wide image.",
                                  limitations="This test cannot establish historical intent.")
    source_lines = PREPROCESSING_SOURCE.splitlines()
    start = next(index for index, line in enumerate(source_lines, 1)
                 if line.startswith("def resize_for_processing("))
    end = next(index for index in range(start + 1, len(source_lines) + 1)
               if source_lines[index - 1].startswith("def ")) - 1
    request = InvestigationRequest(repository_id=repo_id, file_path="src/preprocessing.py", start_line=start,
                                   end_line=end, question="Test the wide image resize behavior.", language="en")
    monkeypatch.setattr("app.investigation_service.complete_json",
                        lambda *args, **kwargs: pytest.fail("the deterministic preprocessing proposal should be used"))
    result = investigate(store, request, test=True, hypothesis=hypothesis)
    assert result["execution_status"] == "NOT_EXECUTED"
    assert result["executed"] is False
    assert "resize_for_processing" in result["test_code"]


@pytest.mark.skipif(os.getenv("CODEKURAL_LIVE_SARVAM") != "1" or not SARVAM_API_KEY,
                    reason="set CODEKURAL_LIVE_SARVAM=1 with SARVAM_API_KEY to run the live provider integration")
def test_live_sarvam_preprocessing_investigation_consistency(tmp_path, monkeypatch):
    store, repo_id, _, _, _ = _collect_preprocessing_evidence(tmp_path, monkeypatch, None)
    request = InvestigationRequest(repository_id=repo_id, file_path="src/preprocessing.py", start_line=1,
                                   end_line=len(PREPROCESSING_SOURCE.splitlines()),
                                   question="Inspect preprocessing behavior and repository test coverage.", language="en")
    # This integration intentionally uses the real complete_json/Sarvam path.
    result = investigate(store, request)
    discovery = result["analysis_scope"]["test_discovery"]
    test_records = {record["test_names"][0]: record for record in discovery["references"]
                    if record["path"] == "tests/test_preprocessing.py" and record["test_names"]}
    assert {name: test_records[name]["classification"] for name in test_records} == {
        "test_prepare_for_detection_returns_grayscale_and_edges": "DIRECT",
        "test_enhance_plate_for_ocr_returns_larger_binary_image": "DIRECT",
        "test_invalid_image_input_is_rejected": "DIRECT",
        "test_blank_image_has_no_plate_candidate": "INDIRECT",
    }
    assert discovery["counts"] == {
        category: sum(item["classification"] == category for item in discovery["references"])
        for category in ("DIRECT", "INDIRECT", "UNRELATED", "UNKNOWN")}
    evidence_ids = {item["id"] for item in result["evidence"]}
    assert all(reference in evidence_ids for item in discovery["references"] for reference in item["evidence_references"])
    for item in result["hypotheses"]:
        assert item["hypothesis_type"] in {"BEHAVIORAL_HYPOTHESIS", "TECHNICAL_HYPOTHESIS"}
        assert item["test_proposal"]["execution_status"] == "NOT_EXECUTED"
        assert item["test_proposal"]["hypothesis_id"] == item["hypothesis_id"]
        assert all(reference in evidence_ids for reference in item["evidence_references"])
        assert all(reference in evidence_ids for reference in item["test_proposal"]["evidence_references"])
    assert len([item for item in result["source_observations"]
                if any(name in item["text"] for name in ("validate_image", "resize_for_processing", "to_grayscale",
                                                          "prepare_for_detection", "enhance_plate_for_ocr"))]) == 5
    documentation_ids = {item["id"] for item in result["evidence"] if item["type"] == "documentation"}
    assert not any(set(item["evidence_references"]) & documentation_ids for item in result["historical_facts"])


def test_preprocessing_test_builder_rejects_unsupported_claim(tmp_path, monkeypatch):
    store, repo_id, evidence, _, _ = _collect_preprocessing_evidence(tmp_path, monkeypatch, "resize_for_processing")
    unsupported = HypothesisRecord(hypothesis_id="HABCDEF12",
                                   hypothesis_text="resize_for_processing preserves the exact aspect ratio for all images.",
                                   hypothesis_type="BEHAVIORAL_HYPOTHESIS", testability=True,
                                   evidence_references=[next(item["id"] for item in evidence if item["type"] == "source_function")],
                                   validation_method="Compare output aspect ratios.", limitations="Integer truncation can alter ratio.")
    assert _build_preprocessing_suggestion(store, repo_id, "src/preprocessing.py", evidence, unsupported) is None


PLATE_DETECTOR_SOURCE = '''from dataclasses import dataclass
import cv2
import numpy as np
from src.preprocessing import prepare_for_detection, resize_for_processing, validate_image

@dataclass(frozen=True)
class DetectionConfig:
    min_area: int = 1_500
    max_area: int = 100_000
    min_aspect_ratio: float = 2.0
    max_aspect_ratio: float = 6.5
    approx_epsilon_ratio: float = 0.02

@dataclass(frozen=True)
class PlateDetection:
    bounding_box: tuple[int, int, int, int]
    contour: np.ndarray
    score: float
    edges: np.ndarray

def _candidate_score(area: float, aspect_ratio: float, config: DetectionConfig) -> float:
    area_score = min(area / config.max_area, 1.0)
    aspect_score = max(0.0, 1.0 - abs(aspect_ratio - 4.0) / 2.5)
    return (0.45 * area_score) + (0.55 * aspect_score)

def detect_license_plate(image: np.ndarray, config: DetectionConfig | None = None) -> PlateDetection | None:
    validate_image(image)
    config = config or DetectionConfig()
    resized, scale = resize_for_processing(image)
    _, edges = prepare_for_detection(resized)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        approximation = cv2.approxPolyDP(contour, config.approx_epsilon_ratio * perimeter, True)
        if len(approximation) != 4:
            continue
        x, y, width, height = cv2.boundingRect(approximation)
        if height == 0:
            continue
        area = width * height
        aspect_ratio = width / height
        if not (config.min_area <= area <= config.max_area and config.min_aspect_ratio <= aspect_ratio <= config.max_aspect_ratio):
            continue
        score = _candidate_score(area, aspect_ratio, config)
        if best is None or score > best[0]:
            best = (score, approximation, (x, y, width, height))
    if best is None:
        return None
    score, contour, (x, y, width, height) = best
    inverse_scale = 1 / scale
    original_box = (int(round(x * inverse_scale)), int(round(y * inverse_scale)),
                    int(round(width * inverse_scale)), int(round(height * inverse_scale)))
    original_contour = np.round(contour.astype(np.float32) * inverse_scale).astype(np.int32)
    return PlateDetection(original_box, original_contour, score, edges)
'''


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)
    return result.stdout


def make_repo(path: Path, files: dict[str, str], message: str = "Add initial source") -> Path:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)
    git(path, "config", "user.email", "test@example.com")
    git(path, "config", "user.name", "CodeKural Test")
    for relative, content in files.items():
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    git(path, "add", ".")
    git(path, "commit", "-m", message)
    return path


def make_test_validation_repo(tmp_path, monkeypatch):
    repo = make_repo(tmp_path / "test_validation_repo", {
        "src/plate_extractor.py": '''import numpy as np
from src.preprocessing import validate_image

def extract_plate(image, bounding_box, padding=6):
    validate_image(image)
    x, y, width, height = bounding_box
    if width <= 0 or height <= 0:
        raise ValueError("Plate bounding box must have positive width and height.")
    image_height, image_width = image.shape[:2]
    left = max(0, x - padding)
    top = max(0, y - padding)
    right = min(image_width, x + width + padding)
    bottom = min(image_height, y + height + padding)
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        raise ValueError("Plate bounding box is outside the image.")
    return crop.copy()
''',
        "src/preprocessing.py": '''import numpy as np

def validate_image(image):
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a NumPy array.")
    if image.size == 0:
        raise ValueError("Image is empty.")
    if image.ndim not in (2, 3):
        raise ValueError("Image must have two or three dimensions.")
    if image.ndim == 3 and image.shape[2] not in (3, 4):
        raise ValueError("Color images must have three or four channels.")
''',
        "requirements.txt": "numpy>=1.26\npytest>=8.0\n",
    })
    store, repo_id = repository_store(monkeypatch, tmp_path, repo)
    _, evidence, _, _ = collect_evidence(store, repo_id, "src/plate_extractor.py", 1, 15, "Suggest tests for selected lines")
    source_function = next(item for item in evidence if item["type"] == "source_function")
    return repo, store, repo_id, source_function


def commit(repo: Path, message: str) -> None:
    git(repo, "add", "-A")
    git(repo, "commit", "-m", message)


def repository_store(monkeypatch, tmp_path, repo):
    monkeypatch.setattr("app.repository_service.WORKSPACE_ROOTS", (tmp_path.resolve(),))
    store = RepositoryStore()
    repo_id = store.validate(str(repo))["repository_id"]
    return store, repo_id


def test_multicommit_line_trace_and_rename_metadata(tmp_path, monkeypatch):
    repo = make_repo(tmp_path / "repo", {"src/old_name.py": "def detector():\n    threshold = 2\n    return threshold\n"})
    (repo / "src" / "old_name.py").write_text("def detector():\n    threshold = 3\n    return threshold\n", encoding="utf-8")
    commit(repo, "Raise detector threshold")
    git(repo, "mv", "src/old_name.py", "src/plate_detector.py")
    (repo / "src" / "plate_detector.py").write_text("def detector():\n    threshold = 4\n    return threshold\n", encoding="utf-8")
    commit(repo, "Rename plate detector and adjust threshold")
    store, repo_id = repository_store(monkeypatch, tmp_path, repo)

    history = file_history(store, repo_id, "src/plate_detector.py", 2, 3)

    assert len(history["commits"]) == 3
    assert history["commits"][0]["parent"]
    assert any(path["status"].startswith("R") for commit_row in history["commits"] for path in commit_row["changed_paths"])
    assert history["line_trace"]
    assert history["blame"]


def test_repository_scan_finds_crossfile_assignments_reads_and_tests(tmp_path, monkeypatch):
    repo = make_repo(tmp_path / "repo", {
        "src/plate.py": "def detect():\n    edges = build_edges()\n    return edges\n",
        "src/graph.py": "def build_edges():\n    edges = []\n    edges.append(1)\n    return edges\n",
        "tests/test_plate.py": "def test_edges():\n    assert detect() == []\n",
    })
    store, repo_id = repository_store(monkeypatch, tmp_path, repo)

    _, evidence, _, scope = collect_evidence(store, repo_id, "src/plate.py", 2, 3)

    assert "edges" in scope["symbols_traced"]
    assert any(item["type"] == "assignment" and "src/graph.py" in item["location"] for item in evidence)
    assert any(item["type"] == "read" and "src/graph.py" in item["location"] for item in evidence)
    assert any(item["type"] == "test" and item["path"] == "tests/test_plate.py" for item in evidence)
    assert scope["complete"] is True


def test_single_root_commit_is_explicitly_insufficient_and_not_intent(tmp_path, monkeypatch):
    repo = make_repo(tmp_path / "repo", {"src/plate_detector.py": "def detect():\n    return edges\n"}, "Initial project snapshot")
    store, repo_id = repository_store(monkeypatch, tmp_path, repo)
    request = InvestigationRequest(repository_id=repo_id, file_path="src/plate_detector.py", start_line=2,
                                   end_line=2, question="Why was this condition added?", language="en")
    _, evidence, _, _ = collect_evidence(store, repo_id, request.file_path, request.start_line, request.end_line)
    source_id = evidence[0]["id"]
    commit_id = next(item["id"] for item in evidence if item["type"] == "commit")

    def fake_complete(system, prompt, output_model, semantic_validator=None, **kwargs):
        assert "Do not claim original intent from a single root commit" in system
        assert "edges" in prompt
        result = output_model.model_validate({
            "summary": "The current return reads the edges value, but the original purpose is unavailable.",
            "summary_evidence_references": [source_id],
            "source_observations": [{"text": "The selected line returns edges.", "evidence_references": [source_id]}],
            "historical_facts": [{"text": "The root commit created the file.", "evidence_references": [commit_id]}],
            "hypotheses": [], "unknowns": [], "contradictions": [],
        })
        if semantic_validator:
            semantic_validator(result)
        return result

    monkeypatch.setattr("app.investigation_service.complete_json", fake_complete)
    result = investigate(store, request)

    assert result["history_status"] == "insufficient"
    assert result["evidence_status"] == "partial"
    assert result["historical_facts"] == []
    assert any("root snapshot" in unknown for unknown in result["unknowns"])
    assert any("author-stated context" in unknown and commit_id in unknown for unknown in result["unknowns"])
    assert result["source_observations"][0]["evidence_references"] == [source_id]


def test_contradiction_is_preserved_with_both_real_evidence_refs(tmp_path, monkeypatch):
    repo = make_repo(tmp_path / "repo", {"src/plate.py": "def detect():\n    return True\n"})
    (repo / "src" / "plate.py").write_text("def detect():\n    return False\n", encoding="utf-8")
    commit(repo, "Fix detector to return true")
    store, repo_id = repository_store(monkeypatch, tmp_path, repo)
    request = InvestigationRequest(repository_id=repo_id, file_path="src/plate.py", start_line=2,
                                   end_line=2, question="What does the change do?", language="en")
    _, evidence, _, _ = collect_evidence(store, repo_id, request.file_path, 2, 2)
    source_id = next(item["id"] for item in evidence if item["type"] == "source")
    diff_id = next(item["id"] for item in evidence if item["type"] == "diff")

    def fake_complete(system, prompt, output_model, semantic_validator=None, **kwargs):
        result = InvestigationOutput.model_validate({
            "summary": "The current return is false while the commit message says it fixes the detector to return true.",
            "summary_evidence_references": [source_id, diff_id],
            "source_observations": [{"text": "The current implementation returns false.", "evidence_references": [source_id]}],
            "historical_facts": [{"text": "The commit message says it fixes the detector to return true.", "evidence_references": [diff_id]}],
            "hypotheses": [], "unknowns": [],
            "contradictions": [{"description": "The commit subject says the fix returns true, but its diff and current source return false.",
                                "why_it_matters": "The message alone does not establish the intended runtime behavior.",
                                "evidence_references": [source_id, diff_id]}],
        })
        semantic_validator(result)
        return result

    monkeypatch.setattr("app.investigation_service.complete_json", fake_complete)
    result = investigate(store, request)

    assert result["contradictions"][0]["evidence_references"] == [source_id, diff_id]
    assert len(result["contradictions"]) == 1


def test_evidence_with_unknown_commit_or_outside_file_is_rejected(tmp_path, monkeypatch):
    repo = make_repo(tmp_path / "repo", {"src/a.py": "value = 1\n"})
    store, repo_id = repository_store(monkeypatch, tmp_path, repo)
    outside = tmp_path / "outside.py"
    outside.write_text("secret = True\n", encoding="utf-8")
    evidence = [{"id": "E01", "type": "source", "location": "../outside.py", "path": "../outside.py",
                 "commit": None, "line_range": "1", "url": None, "excerpt": "secret = True",
                 "relevance_reason": "invalid path"}]

    with pytest.raises(HTTPException) as exc:
        _validate_evidence(store, repo_id, evidence)

    assert exc.value.status_code == 500


def test_fake_commit_reference_is_rejected(tmp_path, monkeypatch):
    repo = make_repo(tmp_path / "repo", {"src/a.py": "value = 1\n"})
    store, repo_id = repository_store(monkeypatch, tmp_path, repo)
    evidence = [{"id": "E01", "type": "commit", "location": "commit:0000000000", "path": "src/a.py",
                 "commit": "0" * 40, "line_range": None, "url": None, "excerpt": "fake commit",
                 "relevance_reason": "invalid commit"}]

    with pytest.raises(HTTPException) as exc:
        _validate_evidence(store, repo_id, evidence)

    assert "commit" in exc.value.detail


def test_commit_evidence_path_must_exist_at_referenced_revision(tmp_path, monkeypatch):
    repo = make_repo(tmp_path / "repo", {"src/a.py": "value = 1\n"})
    store, repo_id = repository_store(monkeypatch, tmp_path, repo)
    commit_id = git(repo, "rev-parse", "HEAD").strip()
    (repo / "src" / "other.py").write_text("value = 2\n", encoding="utf-8")
    evidence = [{"id": "E01", "type": "commit", "category": "B_history", "location": "commit", "path": "src/other.py",
                 "commit": commit_id, "line_range": None, "url": None, "excerpt": "mismatched path",
                 "relevance_reason": "Git commit metadata. Its message is the author's stated intent, not independent proof."}]

    with pytest.raises(HTTPException) as exc:
        _validate_evidence(store, repo_id, evidence)

    assert "does not exist at its referenced commit" in exc.value.detail


def test_scan_finishes_independently_of_evidence_card_count_and_groups_blame(tmp_path, monkeypatch):
    files = {"src/plate.py": "def detect():\n    edges = 1\n    return edges\n"}
    files.update({f"src/module_{index}.py": f"VALUE_{index} = {index}\n" for index in range(24)})
    repo = make_repo(tmp_path / "repo", files)
    store, repo_id = repository_store(monkeypatch, tmp_path, repo)

    _, evidence, _, scope = collect_evidence(store, repo_id, "src/plate.py", 2, 3, "What happens to edges?", 1)

    assert scope["files_inspected"] == scope["files_available"] == 25
    assert len([item for item in evidence if item["type"] == "blame"]) == 1
    assert evidence[0]["category"] == "A_source"
    assert scope["category_counts"]["D_cross_references"] >= 1


def test_test_discovery_inspects_files_and_separates_direct_indirect_and_unrelated(tmp_path, monkeypatch):
    repo = make_repo(tmp_path / "repo", {
        "src/plate_extractor.py": "from src.preprocessing import validate_image\n\ndef extract_plate(image):\n    validate_image(image)\n    return image.copy()\n",
        "src/preprocessing.py": "def validate_image(image):\n    if image is None:\n        raise ValueError('invalid')\n\ndef resize_image(image):\n    return image.copy()\n",
        "tests/test_plate_extractor.py": "import numpy as np\nfrom src.plate_extractor import extract_plate\n\ndef test_extract_plate_returns_copy():\n    image = np.zeros((2, 2))\n    assert extract_plate(image) is not image\n",
        "tests/test_preprocessing.py": "import numpy as np\nimport pytest\nfrom src.preprocessing import resize_image, validate_image\n\ndef test_validate_image_rejects_none():\n    with pytest.raises(ValueError):\n        validate_image(None)\n\ndef test_resize_image_returns_a_copy():\n    image = np.zeros((2, 2))\n    assert resize_image(image) is not image\n",
        "tests/test_ocr.py": "from src.utils import clean_plate_text\n\ndef test_clean_plate_text():\n    assert clean_plate_text(' ab ') == 'AB'\n",
        "src/utils.py": "def clean_plate_text(text):\n    return text.strip().upper()\n",
        "requirements.txt": "numpy>=1.26\npytest>=8.0\n",
        "pytest.ini": "[pytest]\ntestpaths = tests\n",
    })
    store, repo_id = repository_store(monkeypatch, tmp_path, repo)

    _, evidence, _, scope = collect_evidence(store, repo_id, "src/plate_extractor.py", 3, 5,
                                             "Test extract_plate", dependency_depth=1)

    discovery = scope["test_discovery"]
    assert discovery["total_test_files_found"] == 3
    references = discovery["references"]
    assert len({item["test_id"] for item in references}) == len(references) == 4
    direct = [item for item in references if item["classification"] == "DIRECT"]
    indirect = [item for item in references if item["classification"] == "INDIRECT"]
    unrelated = [item for item in references if item["classification"] == "UNRELATED"]
    assert [item["test_names"] for item in direct] == [["test_extract_plate_returns_copy"]]
    assert indirect == []
    assert {item["test_names"][0] for item in unrelated} == {
        "test_clean_plate_text", "test_resize_image_returns_a_copy", "test_validate_image_rejects_none"}
    assert {(item["path"], item["test_names"][0]) for item in unrelated} == {
        ("tests/test_ocr.py", "test_clean_plate_text"),
        ("tests/test_preprocessing.py", "test_resize_image_returns_a_copy"),
        ("tests/test_preprocessing.py", "test_validate_image_rejects_none"),
    }
    assert discovery["counts"] == {"DIRECT": 1, "INDIRECT": 0, "UNRELATED": 3, "UNKNOWN": 0}
    assert discovery["coverage_unknown"] is True
    assert discovery["coverage_status"] == "not_measured"
    assert scope["related_tests_found"] is True
    test_evidence = {item["id"]: item for item in evidence if item["type"] == "test"}
    direct_ref = direct[0]["evidence_references"][0]
    assert direct_ref in test_evidence
    assert test_evidence[direct_ref]["path"] == "tests/test_plate_extractor.py"
    assert any(item["type"] == "dependency_configuration" for item in evidence)
    assert any(item["type"] == "test_configuration" for item in evidence)


def test_discovery_reports_found_but_unrelated_test_files(tmp_path, monkeypatch):
    repo = make_repo(tmp_path / "repo", {
        "src/plate.py": "def extract_plate(image):\n    return image.copy()\n",
        "tests/test_ocr.py": "from src.ocr import normalize\n\ndef test_normalize_text():\n    assert normalize(' ab ') == 'AB'\n",
        "src/ocr.py": "def normalize(value):\n    return value.strip().upper()\n",
    })
    store, repo_id = repository_store(monkeypatch, tmp_path, repo)

    _, _, _, scope = collect_evidence(store, repo_id, "src/plate.py", 1, 2, "extract plate", dependency_depth=1)

    discovery = scope["test_discovery"]
    assert discovery["total_test_files_found"] == 1
    assert discovery["counts"] == {"DIRECT": 0, "INDIRECT": 0, "UNRELATED": 1, "UNKNOWN": 0}
    assert discovery["references"][0]["classification"] == "UNRELATED"
    assert discovery["references"][0]["path"] == "tests/test_ocr.py"
    assert discovery["references"][0]["test_names"] == ["test_normalize_text"]
    assert discovery["coverage_unknown"] is True


def test_broken_discovered_test_file_is_counted_once_as_unknown(tmp_path, monkeypatch):
    repo = make_repo(tmp_path / "repo", {
        "src/plate.py": "def extract_plate(image):\n    return image.copy()\n",
        "tests/test_broken.py": "def test_invalid(:\n    pass\n",
    })
    store, repo_id = repository_store(monkeypatch, tmp_path, repo)
    _, _, _, scope = collect_evidence(store, repo_id, "src/plate.py", 1, 2, "extract plate", dependency_depth=0)
    discovery = scope["test_discovery"]
    assert discovery["counts"] == {"DIRECT": 0, "INDIRECT": 0, "UNRELATED": 0, "UNKNOWN": 1}
    assert discovery["references"][0]["classification"] == "UNKNOWN"
    assert discovery["total_test_files_found"] == 1


def test_duplicate_test_ids_are_rejected_by_consistency_validation():
    item = {"test_id": "tests/test_a.py::test_x#1", "classification": "DIRECT", "evidence_references": []}
    result = {"analysis_scope": {"test_discovery": {
        "references": [item, dict(item)], "counts": {"DIRECT": 2, "INDIRECT": 0, "UNRELATED": 0, "UNKNOWN": 0},
        "coverage_status": "not_measured", "coverage_percent": None}},
        "source_observations": [], "historical_facts": [], "hypotheses": []}
    with pytest.raises(ValueError, match="duplicate test IDs"):
        _validate_investigation_consistency(result, {})


def test_test_counts_must_match_the_canonical_reference_list():
    reference = {"test_id": "tests/test_a.py::test_x#1", "path": "tests/test_a.py", "classification": "DIRECT", "evidence_references": []}
    result = {"analysis_scope": {"test_discovery": {
        "total_test_files_found": 1,
        "references": [reference], "counts": {"DIRECT": 0, "INDIRECT": 0, "UNRELATED": 0, "UNKNOWN": 0},
        "coverage_status": "not_measured", "coverage_percent": None}},
        "source_observations": [], "historical_facts": [], "hypotheses": []}
    with pytest.raises(ValueError, match="DIRECT count"):
        _validate_investigation_consistency(result, {})


def test_documented_module_role_is_not_labeled_historical_intent():
    claim = EvidenceClaim(text="The OCR module owns engine initialization and result handling.",
                          evidence_references=["E01"], hypothesis_type="ARCHITECTURAL_HYPOTHESIS")
    documented = {"E01": {"type": "documentation"}}
    classified = _classify_hypothesis(claim, documented)
    assert classified["hypothesis_type"] == "DOCUMENTED_ARCHITECTURAL_ROLE"
    assert classified["testability"] is False
    history_claim = EvidenceClaim(text="The author introduced the module to reduce startup time.",
                                  evidence_references=["E02"], hypothesis_type="HISTORICAL_HYPOTHESIS")
    assert _classify_hypothesis(history_claim, {"E02": {"type": "commit"}})["hypothesis_type"] == "HISTORICAL_HYPOTHESIS"


def test_overlapping_hypotheses_are_deduplicated():
    documented = EvidenceClaim(text="The OCR module owns engine initialization and result handling.",
                               evidence_references=["E01"], hypothesis_type="DOCUMENTED_ARCHITECTURAL_ROLE")
    overlap = EvidenceClaim(text="OCR module owns initialization and result handling.",
                            evidence_references=["E02"], hypothesis_type="ARCHITECTURAL_HYPOTHESIS")
    unique = EvidenceClaim(text="Returned crops do not share memory with the input image.",
                           evidence_references=["E03"], hypothesis_type="BEHAVIORAL_HYPOTHESIS")
    result = _deduplicate_hypotheses([overlap, documented, unique])
    assert len(result) == 2
    assert result[0] is documented


def test_evidence_is_ranked_and_test_references_are_remapped():
    evidence = [
        {"id": "E01", "type": "documentation", "path": "README.md", "line_range": None},
        {"id": "E02", "type": "test", "path": "tests/test_x.py", "line_range": "2"},
        {"id": "E03", "type": "assignment", "path": "src/caller.py", "line_range": "5"},
        {"id": "E04", "type": "source", "path": "src/target.py", "line_range": "3-5"},
        {"id": "E05", "type": "commit", "path": "src/target.py", "line_range": None},
    ]
    discovery = {"references": [{"evidence_references": ["E02"]}]}
    _rank_evidence(evidence, discovery)
    assert [item["type"] for item in evidence] == ["source", "test", "assignment", "documentation", "commit"]
    test_id = next(item["id"] for item in evidence if item["type"] == "test")
    assert discovery["references"][0]["evidence_references"] == [test_id]


def test_unsupported_claim_evidence_is_rejected():
    with pytest.raises(ValueError, match="no relevant evidence"):
        _validate_claim_support(["E01"], {"E01": {"type": "documentation"}}, {"source", "source_function"}, "source fact")


def test_coverage_is_only_reported_as_measured_when_report_parses(tmp_path, monkeypatch):
    repo = make_repo(tmp_path / "repo", {
        "src/plate.py": "def extract_plate(image):\n    return image.copy()\n",
    })
    store, repo_id = repository_store(monkeypatch, tmp_path, repo)
    assert _coverage_measurement(repo_id, store.context_files(repo_id), store)["coverage_status"] == "not_measured"
    (repo / "coverage.xml").write_text('<coverage line-rate="0.625"/>', encoding="utf-8")
    measured = _coverage_measurement(repo_id, store.context_files(repo_id), store)
    assert measured["coverage_status"] == "measured"
    assert measured["coverage_percent"] == 62.5
    assert measured["coverage_report_path"] == "coverage.xml"


def test_rationale_and_defect_hypotheses_need_concrete_support():
    source_only = {"E01": {"type": "source"}}
    docs = {"E02": {"type": "documentation"}}
    failing_test = {"E03": {"type": "test"}}

    assert not _hypothesis_has_support(EvidenceClaim(text="The modules have separate responsibilities.", evidence_references=["E01"],
                                                     hypothesis_type="ARCHITECTURAL_HYPOTHESIS"), source_only)
    assert not _hypothesis_has_support(EvidenceClaim(text="Docs state the reason.", evidence_references=["E02"],
                                                     hypothesis_type="HISTORICAL_HYPOTHESIS"), docs)
    history = {"E04": {"type": "commit"}}
    assert _hypothesis_has_support(EvidenceClaim(text="The author stated the reason in the commit.", evidence_references=["E04"],
                                                 hypothesis_type="HISTORICAL_HYPOTHESIS"), history)
    assert _hypothesis_has_support(EvidenceClaim(text="This input produces an empty crop and raises ValueError.", evidence_references=["E01"],
                                                 hypothesis_type="BEHAVIORAL_HYPOTHESIS"), source_only)
    assert _hypothesis_has_support(EvidenceClaim(text="The assertion demonstrates this behavior.", evidence_references=["E03"],
                                                 hypothesis_type="BEHAVIORAL_HYPOTHESIS"), failing_test)


@pytest.mark.parametrize(("legacy_type", "expected_type", "testable"), [
    ("BEHAVIORAL_HYPOTHESIS", "BEHAVIORAL_HYPOTHESIS", True),
    ("TECHNICAL_HYPOTHESIS", "TECHNICAL_HYPOTHESIS", True),
    ("ARCHITECTURAL_HYPOTHESIS", "ARCHITECTURAL_HYPOTHESIS", False),
    ("HISTORICAL_HYPOTHESIS", "HISTORICAL_HYPOTHESIS", False),
])
def test_hypotheses_are_classified_into_v23_taxonomy(legacy_type, expected_type, testable):
    claim = EvidenceClaim(text="The selected implementation has a grounded implication.",
                          evidence_references=["E01"], hypothesis_type=legacy_type)
    classified = HypothesisRecord.model_validate(_classify_hypothesis(claim))
    assert classified.hypothesis_type == expected_type
    assert classified.testability is testable
    assert classified.hypothesis_id.startswith("H")
    assert classified.evidence_references == ["E01"]
    if not testable:
        assert "cannot be established through a regression test" in classified.limitations


def test_architectural_hypothesis_requires_documentation_or_tests():
    source_only = {"E01": {"type": "source"}}
    documented = {"E01": {"type": "documentation"}}
    claim = EvidenceClaim(text="The modules have separate responsibilities.", evidence_references=["E01"],
                          hypothesis_type="ARCHITECTURAL_HYPOTHESIS")
    assert not _hypothesis_has_support(claim, source_only)
    assert _hypothesis_has_support(claim, documented)


def test_source_fact_is_not_returned_as_a_hypothesis():
    evidence = {"E01": {"type": "source_function", "excerpt": "def extract_plate(image):\n    return crop.copy()\n"}}
    fact = EvidenceClaim(text="The function returns crop.copy().", evidence_references=["E01"],
                         hypothesis_type="TECHNICAL_HYPOTHESIS")
    behavioral = EvidenceClaim(text="Mutating the returned crop leaves the input unchanged.", evidence_references=["E01"],
                               hypothesis_type="BEHAVIORAL_HYPOTHESIS")
    facts, hypotheses = _separate_source_facts([fact, behavioral], [], evidence)
    assert [claim.text for claim in facts] == ["The function returns crop.copy()."]
    assert facts[0].hypothesis_type == "TECHNICAL_HYPOTHESIS"  # promoted to a SOURCE_FACT by the output serializer
    assert [claim.hypothesis_type for claim in hypotheses] == ["BEHAVIORAL_HYPOTHESIS"]
    facts, remaining = _separate_source_facts([fact], facts, evidence)
    assert len(facts) == 1
    assert not remaining


def test_documentation_role_stays_distinct_from_historical_motivation():
    docs = {"E01": {"type": "documentation"}}
    documented = EvidenceClaim(text="README describes configurable area and aspect-ratio filtering.",
                               evidence_references=["E01"], hypothesis_type="ARCHITECTURAL_HYPOTHESIS")
    assert _classify_hypothesis(documented, docs)["hypothesis_type"] == "DOCUMENTED_ARCHITECTURAL_ROLE"

    historical_text = "These thresholds were selected during the initial implementation to filter plate candidates."
    historical = EvidenceClaim(text=historical_text, evidence_references=["E01"],
                               hypothesis_type="ARCHITECTURAL_HYPOTHESIS")
    assert _mentions_historical_motivation(historical_text)
    assert not _hypothesis_has_support(historical, docs)
    assert _classify_hypothesis(historical, docs)["hypothesis_type"] == "HISTORICAL_HYPOTHESIS"


def test_documented_claim_is_validated_against_documentation_not_source():
    documented = EvidenceClaim(text="README describes configurable plate filters.", evidence_references=["E01"],
                               hypothesis_type="DOCUMENTED_ARCHITECTURAL_ROLE")
    output = InvestigationOutput(summary="The README documents configurable filters.",
                                 summary_evidence_references=["E01"], source_observations=[documented],
                                 historical_facts=[], hypotheses=[], unknowns=[], contradictions=[])
    _validate_investigation_references(output, {"E01": {"type": "documentation"}})
    with pytest.raises(ValueError, match="no relevant evidence"):
        _validate_investigation_references(
            output.model_copy(update={"source_observations": [documented.model_copy(update={"hypothesis_type": None})]}),
            {"E01": {"type": "documentation"}},
        )


def test_documentation_claim_moves_to_documented_role_and_root_snapshot_cannot_support_history(tmp_path, monkeypatch):
    repo = make_repo(tmp_path / "repo", {
        "src/plate_detector.py": "class DetectionConfig:\n    min_area: int = 1500\n    max_area: int = 100000\n",
        "README.md": "Configurable area and aspect-ratio filters are supported.\n",
    })
    store, repo_id = repository_store(monkeypatch, tmp_path, repo)
    request = InvestigationRequest(repository_id=repo_id, file_path="src/plate_detector.py", start_line=1,
                                   end_line=3, question="Why were these thresholds selected?", language="en")
    _, evidence, _, _ = collect_evidence(store, repo_id, request.file_path, 1, 3, request.question)
    doc_id = next(item["id"] for item in evidence if item["type"] == "documentation")
    commit_id = next(item["id"] for item in evidence if item["type"] == "commit")

    def fake_complete(system, prompt, output_model, semantic_validator=None, **kwargs):
        assert "Treat them as investigation targets only" in system
        assert "Source-derived technical cues (not conclusions)" in prompt
        assert "configuration defaults min_area=1500, max_area=100000" in prompt
        result = InvestigationOutput.model_validate({
            "summary": "The README documents configurable filters, while the threshold rationale is unknown.",
            "summary_evidence_references": ["E01"],
            "source_observations": [{"text": "README documents configurable filters.",
                                     "evidence_references": [doc_id],
                                     "hypothesis_type": "DOCUMENTED_ARCHITECTURAL_ROLE"}],
            "historical_facts": [],
            "hypotheses": [{"text": "The thresholds were selected from the outset to match the author's initial vision.",
                            "evidence_references": [commit_id, doc_id],
                            "hypothesis_type": "HISTORICAL_HYPOTHESIS"}],
            "unknowns": [], "contradictions": [],
        })
        semantic_validator(result)
        return result

    monkeypatch.setattr("app.investigation_service.complete_json", fake_complete)
    result = investigate(store, request)

    assert result["source_observations"] == []
    assert len(result["hypotheses"]) == 1
    assert result["hypotheses"][0]["hypothesis_type"] == "DOCUMENTED_ARCHITECTURAL_ROLE"
    assert not any("selected from the outset" in item["hypothesis_text"] for item in result["hypotheses"])
    assert any("Historical motivation remains unverified" in unknown for unknown in result["unknowns"])


def test_python_technical_cues_expose_thresholds_formula_and_predicates():
    excerpt = '''from dataclasses import dataclass
@dataclass(frozen=True)
class DetectionConfig:
    min_area: int = 1_500
    max_area: int = 100_000
    min_aspect_ratio: float = 2.0
    max_aspect_ratio: float = 6.5
def _candidate_score(area, aspect_ratio, config):
    area_score = min(area / config.max_area, 1.0)
    aspect_score = max(0.0, 1.0 - abs(aspect_ratio - 4.0) / 2.5)
    return (0.45 * area_score) + (0.55 * aspect_score)
def detect(area, ratio, config):
    if not (config.min_area <= area <= config.max_area and config.min_aspect_ratio <= ratio <= config.max_aspect_ratio):
        return None
    return _candidate_score(area, ratio, config)
'''
    cues = _python_technical_cues([{"id": "E06", "type": "source_function", "path": "src/plate_detector.py",
                                    "excerpt": excerpt}])
    assert "DetectionConfig" in cues
    assert "min_area=1500" in cues
    assert "_candidate_score" in cues
    assert "area / config.max_area" in cues
    assert "aspect_ratio - 4.0" in cues
    assert "config.min_area <= area <= config.max_area" in cues


def _collect_candidate_score_evidence(tmp_path, monkeypatch):
    source = PLATE_DETECTOR_SOURCE
    repo = make_repo(tmp_path / "detector_repo", {
        "src/__init__.py": "",
        "src/plate_detector.py": source,
        "src/preprocessing.py": "def prepare_for_detection(image):\n    return image, image\ndef resize_for_processing(image):\n    return image, 1.0\ndef validate_image(image):\n    return None\n",
        "tests/test_detector.py": "from src.plate_detector import detect_license_plate\ndef test_detector_returns_none_for_blank_image():\n    assert detect_license_plate(None) is None\n",
        "requirements.txt": "pytest>=8.0\nnumpy>=1.26\n",
        "README.md": "The detector scores candidates using area and aspect ratio.\n",
    })
    store, repo_id = repository_store(monkeypatch, tmp_path, repo)
    score_line = next(index for index, line in enumerate(source.splitlines(), 1)
                      if line.startswith("def _candidate_score"))
    _, evidence, _, _ = collect_evidence(store, repo_id, "src/plate_detector.py", score_line,
                                         score_line + 3, "Investigate candidate score", dependency_depth=2)
    return store, repo_id, evidence


def test_candidate_score_formula_hypotheses_and_pipeline_evidence(tmp_path, monkeypatch):
    _, _, evidence = _collect_candidate_score_evidence(tmp_path, monkeypatch)
    contract = _candidate_score_contract(evidence)
    assert contract is not None
    assert contract["config_values"]["max_area"] == 100_000
    assert contract["config_values"]["min_area"] == 1_500
    assert contract["area_cap"] == 1
    assert contract["aspect_floor"] == 0
    assert contract["aspect_center"] == 4
    assert contract["aspect_decay"] == 2.5
    assert contract["area_weight"] == 0.45
    assert contract["aspect_weight"] == 0.55

    claims = _derive_candidate_score_hypotheses(evidence)
    assert len(claims) == 5
    assert all(claim.hypothesis_type == "TECHNICAL_HYPOTHESIS" for claim in claims)
    assert all(claim.evidence_references and set(claim.evidence_references) <= {item["id"] for item in evidence}
               for claim in claims)
    assert len({claim.text for claim in claims}) == len(claims)

    functions = [item["excerpt"] for item in evidence
                 if item["type"] in {"source_function", "related_function"}]
    detector_bodies = "\n".join(functions)
    assert "def detect_license_plate" in detector_bodies
    for call_path in ("resize_for_processing", "prepare_for_detection", "cv2.findContours", "cv2.approxPolyDP",
                      "cv2.boundingRect", "_candidate_score", "PlateDetection"):
        assert call_path in detector_bodies
    assert any("min_area <= area <= config.max_area" in excerpt for excerpt in functions)


def test_candidate_score_proposals_are_complete_ast_valid_and_not_executed(tmp_path, monkeypatch):
    store, repo_id, evidence = _collect_candidate_score_evidence(tmp_path, monkeypatch)
    by_id = {item["id"]: item for item in evidence}
    claims = _derive_candidate_score_hypotheses(evidence)
    assert len(claims) == 5
    for index, claim in enumerate(claims):
        hypothesis = HypothesisRecord(hypothesis_id=f"H{index + 1:08X}", hypothesis_text=claim.text,
                                      hypothesis_type=claim.hypothesis_type, testability=True,
                                      evidence_references=claim.evidence_references,
                                      validation_method="Compare deterministic _candidate_score outputs to the source formula.",
                                      limitations="Does not exercise contour generation or establish historical intent.")
        suggestion = _build_candidate_score_suggestion(store, repo_id, "src/plate_detector.py", evidence, hypothesis)
        assert suggestion is not None, f"No complete supported proposal for hypothesis: {claim.text}"
        tree = ast.parse(suggestion.test_code)
        assert any(node.name.startswith("test_") for node in tree.body if isinstance(node, ast.FunctionDef))
        assert "_candidate_score" in suggestion.test_code
        assert "assert " in suggestion.test_code
        assert suggestion.execution_status == "NOT_EXECUTED"
        assert set(suggestion.evidence_references) <= set(by_id)
        assert set(claim.evidence_references) <= set(suggestion.evidence_references)
        assert "pytest.approx(" in suggestion.test_code


def test_documentation_evidence_cannot_be_mislabeled_as_git_diff(tmp_path, monkeypatch):
    store, repo_id, evidence = _collect_candidate_score_evidence(tmp_path, monkeypatch)
    documentation = next(item for item in evidence if item["type"] == "documentation")
    invalid = dict(documentation, category="B_history", relevance_reason="Git before-and-after diff from the current worktree.")
    with pytest.raises(HTTPException, match="type/category mismatch"):
        _validate_evidence(store, repo_id, [invalid])
    invalid = dict(documentation, relevance_reason="Git before-and-after diff from the current worktree.")
    with pytest.raises(HTTPException, match="description does not match its type"):
        _validate_evidence(store, repo_id, [invalid])


def test_test_relationships_follow_transitive_project_dependencies(tmp_path, monkeypatch):
    repo = make_repo(tmp_path / "repo", {
        "src/target.py": "from src.helpers import first\ndef target(value):\n    return first(value)\n",
        "src/helpers.py": "from src.normalizer import normalize\ndef first(value):\n    return normalize(value)\n",
        "src/normalizer.py": "def normalize(value):\n    return value.strip()\n",
        "src/caller.py": "from src.target import target\ndef run(value):\n    return target(value)\n",
        "tests/test_relationships.py": "from src.caller import run\nfrom src.normalizer import normalize\nfrom src.target import target\ndef test_direct():\n    assert target(' x ') == 'x'\ndef test_transitive_dependency():\n    assert normalize(' x ') == 'x'\ndef test_transitive_caller():\n    assert run(' x ') == 'x'\ndef test_import_only():\n    assert callable(target)\n",
    })
    store, repo_id = repository_store(monkeypatch, tmp_path, repo)

    _, _, _, scope = collect_evidence(store, repo_id, "src/target.py", 2, 3,
                                      "Check target behavior", dependency_depth=3)
    refs = {item["test_names"][0]: item["classification"] for item in scope["test_discovery"]["references"]}
    assert refs == {"test_direct": "DIRECT", "test_transitive_dependency": "UNRELATED",
                    "test_transitive_caller": "INDIRECT", "test_import_only": "UNRELATED"}


def test_unknown_evidence_references_are_rejected():
    with pytest.raises(ValueError, match="unknown evidence IDs: E99"):
        _validate_references(["E99"], {"E01": {"type": "source"}})


def test_extract_plate_test_proposal_covers_actual_source_contract(tmp_path, monkeypatch):
    _, store, repo_id, source_function = make_test_validation_repo(tmp_path, monkeypatch)
    code = _extract_plate_test_code()
    _validate_test_code(store, repo_id, code, "pytest>=8.0\nnumpy>=1.26\n", {"extract_plate"}, [source_function])
    assert "@pytest.mark.parametrize(\"width\", [0, -1])" in code
    assert "@pytest.mark.parametrize(\"height\", [0, -1])" in code
    assert "partially_outside_each_image_edge" in code
    assert "padding_near_edge_clips_each_bound" in code
    assert "three or four channels" in code
    assert "NOT_EXECUTED" not in code  # status belongs to the response, not executable source


def test_hypothesis_specific_copy_proposal_contains_only_copy_case():
    code = _extract_plate_test_code("Mutating the returned crop does not change the input because it is an independent copy.")
    assert "test_returned_crop_is_an_independent_copy" in code
    assert "test_zero_width" not in code
    assert "test_padding_near_edge" not in code


def test_evidence_labels_are_normalized_to_bare_ids_before_validation():
    assert _normalize_test_references(["E01: src/plate_extractor.py line 10", "E02 — README.md", "E01"]) == ["E01", "E02"]


def test_complete_extract_plate_test_code_passes_static_quality_checks(tmp_path, monkeypatch):
    _, store, repo_id, source_function = make_test_validation_repo(tmp_path, monkeypatch)
    manifests = "pytest>=8.0\nnumpy>=1.26\n"

    _validate_test_code(store, repo_id, PLATE_EXTRACTOR_TEST_CODE, manifests,
                        {"extract_plate"}, [source_function])


def test_extract_plate_suggestion_is_complete_cited_and_not_executed(tmp_path, monkeypatch):
    _, store, repo_id, _ = make_test_validation_repo(tmp_path, monkeypatch)
    _, evidence, _, _ = collect_evidence(store, repo_id, "src/plate_extractor.py", 1, 15, "Suggest extract_plate tests")

    result = _build_extract_plate_suggestion(store, repo_id, "src/plate_extractor.py", evidence)

    assert result.test_framework == "pytest with NumPy"
    assert result.execution_status == "NOT_EXECUTED"
    assert len(result.evidence_references) >= 3
    assert {item.test_type for item in result.test_objectives} == {
        "behavioral_regression", "hypothesis_validation", "historical_intent_verification"}
    assert not next(item for item in result.test_objectives if item.test_type == "historical_intent_verification").testable
    assert "def test_valid_crop_returns_expected_pixels" in result.test_code
    assert "def test_completely_outside_right_or_bottom_raises_value_error" in result.test_code
    assert "def test_returned_crop_is_an_independent_copy" in result.test_code


def test_extract_plate_suggestion_matches_selected_hypothesis(tmp_path, monkeypatch):
    _, store, repo_id, source_function = make_test_validation_repo(tmp_path, monkeypatch)
    _, evidence, _, _ = collect_evidence(store, repo_id, "src/plate_extractor.py", 1, 15, "copy independence")
    hypothesis = HypothesisRecord(
        hypothesis_id="H1234ABCD",
        hypothesis_text="Mutating the returned crop does not change the input because the crop is an independent copy.",
        hypothesis_type="BEHAVIORAL_HYPOTHESIS",
        testability=True,
        evidence_references=[source_function["id"]],
        validation_method="Mutate the returned crop and compare the input array with a saved copy.",
        limitations="This checks array storage for this crop only.",
    )

    result = _build_extract_plate_suggestion(store, repo_id, "src/plate_extractor.py", evidence, hypothesis)

    assert "test_returned_crop_is_an_independent_copy" in result.test_code
    assert "test_zero_width" not in result.test_code
    assert "does not mutate or share storage" in result.expected_behavior
    assert result.hypothesis_tested == hypothesis.hypothesis_text
    assert result.hypothesis_id == hypothesis.hypothesis_id
    assert result.execution_status == "NOT_EXECUTED"


@pytest.mark.parametrize(("code", "error"), [
        (PLATE_EXTRACTOR_TEST_CODE.replace("(2, 1, 3, 2)", "(2, 1, missing_box, 2)", 1), "undefined names"),
    (PLATE_EXTRACTOR_TEST_CODE.replace("import numpy as np\n", "", 1), "undefined names"),
        (PLATE_EXTRACTOR_TEST_CODE.replace("    assert np.array_equal(crop, expected)\n", "    pass\n", 1)
         .replace("    assert crop.shape == (2, 3)\n", "", 1), "no assertion"),
        (PLATE_EXTRACTOR_TEST_CODE.replace("    assert crop.shape == (2, 3)", "    assert True", 1), "incomplete assert True"),
        (PLATE_EXTRACTOR_TEST_CODE.replace("padding=0", "padding_made_up=0", 1), "unknown keyword arguments"),
        (PLATE_EXTRACTOR_TEST_CODE.replace("import pytest\n", "import pytest\nimport undeclared_package\n", 1), "neither standard-library nor declared"),
    (PLATE_EXTRACTOR_TEST_CODE.replace("extract_plate", "missing_function"), "imports missing names"),
        (PLATE_EXTRACTOR_TEST_CODE.replace("test_zero_width_and_negative_width_are_rejected", "test_invalid_width_box_is_rejected"), "missing cases"),
])
def test_invalid_generated_test_code_is_rejected(tmp_path, monkeypatch, code, error):
    _, store, repo_id, source_function = make_test_validation_repo(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match=error):
        _validate_test_code(store, repo_id, code, "pytest>=8.0\nnumpy>=1.26\n", {"extract_plate"},
                            [source_function])


def test_missing_history_and_tests_are_reported_without_claiming_origin(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    source = repo / "src" / "plate.py"
    source.parent.mkdir()
    source.write_text("def detect():\n    return value\n", encoding="utf-8")
    store, repo_id = repository_store(monkeypatch, tmp_path, repo)
    request = InvestigationRequest(repository_id=repo_id, file_path="src/plate.py", start_line=2,
                                   end_line=2, question="Why does this return value?", language="en")
    _, evidence, history, scope = collect_evidence(store, repo_id, request.file_path, 2, 2)
    assert history["commits"] == []
    assert not any(item["type"] == "test" for item in evidence)
    assert scope["files_available"] == 1

    def fake_complete(system, prompt, output_model, semantic_validator=None, **kwargs):
        result = InvestigationOutput.model_validate({
            "summary": "The code returns the current value, but its original reason is unknown.",
            "summary_evidence_references": ["E01"],
            "source_observations": [{"text": "The selected function returns value.", "evidence_references": ["E01"]}],
            "historical_facts": [], "hypotheses": [], "unknowns": [], "contradictions": [],
        })
        semantic_validator(result)
        return result

    monkeypatch.setattr("app.investigation_service.complete_json", fake_complete)
    result = investigate(store, request)

    assert result["history_status"] == "missing"
    assert any("No Git commits" in unknown for unknown in result["unknowns"])
    assert result["evidence_status"] == "partial"


def test_tamil_prompt_keeps_identifiers_and_context_scope(tmp_path, monkeypatch):
    repo = make_repo(tmp_path / "repo", {"src/plate.py": "def detect_plate():\n    edges = build_edges()\n    return edges\n"})
    store, repo_id = repository_store(monkeypatch, tmp_path, repo)
    request = InvestigationRequest(repository_id=repo_id, file_path="src/plate.py", start_line=2,
                                   end_line=3, question="What happens to edges?", language="ta")
    prompts = {}

    def fake_complete(system, prompt, output_model, semantic_validator=None, **kwargs):
        prompts["system"] = system
        prompts["prompt"] = prompt
        result = InvestigationOutput.model_validate({
            "summary": "`edges` is assigned from `build_edges()` and returned.",
            "summary_evidence_references": ["E01"],
            "source_observations": [{"text": "`edges` receives `build_edges()` output.", "evidence_references": ["E01"]}],
            "historical_facts": [], "hypotheses": [], "unknowns": [], "contradictions": [],
        })
        semantic_validator(result)
        return result

    monkeypatch.setattr("app.investigation_service.complete_json", fake_complete)
    result = investigate(store, request)

    assert "Answer in Tamil" in prompts["prompt"]
    assert "Preserve technical identifiers" in prompts["prompt"]
    assert "build_edges" in prompts["prompt"]
    assert result["source_observations"]


def test_test_generation_contract_uses_evidence_references_and_not_executed(tmp_path, monkeypatch):
    repo = make_repo(tmp_path / "repo", {
        "src/plate.py": "def detect():\n    return False\n",
        "requirements.txt": "pytest>=8\nnumpy>=1.26\n",
        "tests/test_plate.py": "def test_placeholder():\n    assert True\n",
    })
    store, repo_id = repository_store(monkeypatch, tmp_path, repo)
    request = InvestigationRequest(repository_id=repo_id, file_path="src/plate.py", start_line=2,
                                   end_line=2, question="Protect this result", language="en")
    _, raw_evidence, _, _ = collect_evidence(store, repo_id, request.file_path, 2, 2, request.question)
    source_id = next(item["id"] for item in raw_evidence if item["type"] == "source_function")
    config_id = next(item["id"] for item in raw_evidence if item["type"] == "dependency_configuration")
    history_id = next(item["id"] for item in raw_evidence if item["type"] == "commit")
    hypothesis = HypothesisRecord(hypothesis_id="H1234ABCD", hypothesis_text="detect returns False for this input.",
                                  hypothesis_type="BEHAVIORAL_HYPOTHESIS", testability=True,
                                  evidence_references=[source_id], validation_method="Assert the return value.",
                                  limitations="Only this input is covered.")

    def fake_complete(system, prompt, output_model, semantic_validator=None, **kwargs):
        assert "evidence_references" in system
        assert "complete executable test functions" in system
        assert hypothesis.hypothesis_id in prompt
        result = SuggestionOutput.model_validate({
            "purpose": "Guard the false return behavior.",
            "test_framework": "pytest",
            "test_code": "from src.plate import detect\n\ndef test_detect_false():\n    assert detect() is False",
            "expected_behavior": "The function returns false.",
            "limitations": "No fixture or framework convention was found.",
            "test_type": "behavioral_regression",
            "hypothesis_id": hypothesis.hypothesis_id,
            "hypothesis_type": hypothesis.hypothesis_type,
            "hypothesis_tested": hypothesis.hypothesis_text,
            "evidence_references": [source_id, history_id],
            "test_objectives": [
                {"test_type": "behavioral_regression", "description": "Protect the current return value.", "testable": True, "evidence_references": [source_id, config_id]},
                {"test_type": "hypothesis_validation", "description": "Check the selected behavior hypothesis.", "testable": True, "evidence_references": [source_id]},
                {"test_type": "historical_intent_verification", "description": "Requires history, documentation, or author confirmation.", "testable": False, "evidence_references": [history_id]},
            ],
            "execution_status": "NOT_EXECUTED",
        })
        semantic_validator(result)
        return result

    monkeypatch.setattr("app.investigation_service.complete_json", fake_complete)
    result = investigate(store, request, test=True, hypothesis=hypothesis)

    assert set(result["evidence_references"]) == {source_id, config_id, history_id}
    assert result["test_framework"] == "pytest"
    assert result["executed"] is False
    assert result["execution_status"] == "NOT_EXECUTED"
    assert result["hypothesis_id"] == hypothesis.hypothesis_id


def test_historical_hypothesis_is_rejected_for_regression_test_generation(tmp_path, monkeypatch):
    repo = make_repo(tmp_path / "repo", {
        "src/plate.py": "def detect():\n    return False\n",
        "requirements.txt": "pytest>=8\n",
    })
    store, repo_id = repository_store(monkeypatch, tmp_path, repo)
    request = InvestigationRequest(repository_id=repo_id, file_path="src/plate.py", start_line=2,
                                   end_line=2, question="What was the original reason?", language="en")
    _, evidence, _, _ = collect_evidence(store, repo_id, request.file_path, 2, 2, request.question)
    source_id = next(item["id"] for item in evidence if item["type"] == "source_function")
    hypothesis = HypothesisRecord(hypothesis_id="H9876ABCD", hypothesis_text="The author selected False for a historical reason.",
                                  hypothesis_type="HISTORICAL_HYPOTHESIS", testability=False,
                                  evidence_references=[source_id], validation_method="Search history and docs.",
                                  limitations="This hypothesis cannot be established through a regression test. Additional historical or design evidence is required.")

    with pytest.raises(HTTPException, match="cannot be established through a regression test"):
        investigate(store, request, test=True, hypothesis=hypothesis)
    with pytest.raises(HTTPException, match="Select a supported behavioral or technical hypothesis"):
        investigate(store, request, test=True)
