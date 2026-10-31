import subprocess
from pathlib import Path

import pytest
from fastapi import HTTPException

from app.repository_service import RepositoryStore


def make_repo(path: Path) -> Path:
    path.mkdir()
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)
    (path / "sample.py").write_text("def retry():\n    return 3\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "sample.py"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-m", "Add retry helper"], check=True, capture_output=True)
    return path


def test_validates_repo_inside_configured_root(tmp_path, monkeypatch):
    repo = make_repo(tmp_path / "repo")
    monkeypatch.setattr("app.repository_service.WORKSPACE_ROOTS", (tmp_path.resolve(),))
    store = RepositoryStore()
    result = store.validate(str(repo))
    assert result["name"] == "repo"
    assert "sample.py" in [item["path"] for item in store.files(result["repository_id"])]


def test_rejects_path_outside_configured_root(tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    repo = make_repo(tmp_path / "outside")
    monkeypatch.setattr("app.repository_service.WORKSPACE_ROOTS", (allowed.resolve(),))
    with pytest.raises(HTTPException) as exc:
        RepositoryStore().validate(str(repo))
    assert exc.value.status_code == 403


def test_rejects_traversal_and_symlink_escape(tmp_path, monkeypatch):
    repo = make_repo(tmp_path / "repo")
    outside = tmp_path / "outside.py"
    outside.write_text("secret", encoding="utf-8")
    try:
        (repo / "escape.py").symlink_to(outside)
    except OSError:
        pytest.skip("Symlink creation is unavailable on this Windows account.")
    monkeypatch.setattr("app.repository_service.WORKSPACE_ROOTS", (tmp_path.resolve(),))
    store = RepositoryStore()
    repo_id = store.validate(str(repo))["repository_id"]
    for candidate in ("../outside.py", "escape.py"):
        with pytest.raises(HTTPException):
            store.resolve_file(repo_id, candidate)
