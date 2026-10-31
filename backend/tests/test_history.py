import subprocess
from pathlib import Path

from app.repository_service import RepositoryStore
from app.git_service import file_history


def test_reads_real_commit_and_blame(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    (repo / "a.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "a.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "Set initial value"], check=True, capture_output=True)
    monkeypatch.setattr("app.repository_service.WORKSPACE_ROOTS", (tmp_path.resolve(),))
    store = RepositoryStore()
    repo_id = store.validate(str(repo))["repository_id"]
    history = file_history(store, repo_id, "a.py", 1, 1)
    assert history["commits"][0]["subject"] == "Set initial value"
    assert history["blame"]


def test_handles_repository_with_no_commits(tmp_path, monkeypatch):
    repo = tmp_path / "unborn"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    (repo / "new.py").write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setattr("app.repository_service.WORKSPACE_ROOTS", (tmp_path.resolve(),))
    store = RepositoryStore()
    repo_id = store.validate(str(repo))["repository_id"]
    history = file_history(store, repo_id, "new.py")
    assert history["commits"] == []
    assert history["shallow"] is False
