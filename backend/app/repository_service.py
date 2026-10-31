import hashlib
import os
import subprocess
from pathlib import Path

from fastapi import HTTPException

from .config import WORKSPACE_ROOTS

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".java", ".c", ".h", ".cc", ".cpp", ".hpp",
}
EXCLUDED_PARTS = {".git", "node_modules", ".venv", "venv", "dist", "build", "coverage", "__pycache__", ".next", "vendor"}
MAX_FILE_BYTES = 250_000


class RepositoryStore:
    def __init__(self) -> None:
        self._repositories: dict[str, Path] = {}

    def validate(self, raw_path: str) -> dict:
        return self._validate(raw_path, require_configured_root=True)

    def validate_selected(self, raw_path: str) -> dict:
        """Validate a folder explicitly chosen in the local OS folder picker."""
        return self._validate(raw_path, require_configured_root=False)

    def _validate(self, raw_path: str, *, require_configured_root: bool) -> dict:
        candidate = Path(raw_path).expanduser()
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            raise HTTPException(400, "That repository path does not exist or cannot be accessed.")
        if require_configured_root and not any(_is_relative_to(resolved, root) for root in WORKSPACE_ROOTS):
            raise HTTPException(403, "Choose a repository inside a configured WORKSPACE_ROOT.")
        if not resolved.is_dir() or not (resolved / ".git").exists():
            raise HTTPException(400, "The selected directory is not a Git repository.")
        default_ref = run_git(resolved, ["symbolic-ref", "--short", "refs/remotes/origin/HEAD"], check=False).strip()
        current_branch = run_git(resolved, ["branch", "--show-current"], check=False).strip()
        repo_id = hashlib.sha256(os.fsencode(str(resolved))).hexdigest()[:20]
        self._repositories[repo_id] = resolved
        default_branch = default_ref.removeprefix("origin/") if default_ref.startswith("origin/") else (default_ref or None)
        return {"repository_id": repo_id, "name": resolved.name, "default_branch": default_branch,
                "current_branch": current_branch or None,
                "supported_file_types": sorted(SUPPORTED_EXTENSIONS)}

    def browse(self, raw_path: str | None = None, search: str = "") -> dict:
        """List local directories or find Git repositories, bounded to configured roots."""
        query = search.strip().casefold()
        if query:
            matches: list[dict] = []
            visited = 0
            for root in WORKSPACE_ROOTS:
                if root.is_dir():
                    for current, dirs, _ in os.walk(root, followlinks=False):
                        current_path = Path(current)
                        depth = len(current_path.relative_to(root).parts)
                        dirs[:] = sorted(name for name in dirs if name not in EXCLUDED_PARTS
                                         and not name.startswith(".") and depth < 8)
                        visited += 1
                        if visited > 20_000:
                            break
                        if (current_path / ".git").exists():
                            if query in current_path.name.casefold() or query in str(current_path).casefold():
                                matches.append({"name": current_path.name, "path": str(current_path),
                                                "is_repository": True})
                                dirs[:] = []
                                if len(matches) >= 100:
                                    break
            return {"roots": [str(root) for root in WORKSPACE_ROOTS], "current_path": None,
                    "entries": sorted(matches, key=lambda item: item["path"].casefold()), "search": True}

        if raw_path is None:
            entries = []
            for root in WORKSPACE_ROOTS:
                if root.is_dir():
                    entries.append({"name": root.name or str(root), "path": str(root),
                                    "is_repository": (root / ".git").exists()})
            return {"roots": [str(root) for root in WORKSPACE_ROOTS], "current_path": None,
                    "entries": entries, "search": False}

        try:
            current = Path(raw_path).resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            raise HTTPException(400, "That folder does not exist or cannot be accessed.")
        root = next((allowed for allowed in WORKSPACE_ROOTS if _is_relative_to(current, allowed)), None)
        if root is None or not current.is_dir():
            raise HTTPException(403, "Choose a folder inside a configured WORKSPACE_ROOT.")
        entries = []
        for child in sorted(current.iterdir(), key=lambda item: item.name.casefold()):
            if child.name in EXCLUDED_PARTS or child.name.startswith(".") or not child.is_dir() or child.is_symlink():
                continue
            entries.append({"name": child.name, "path": str(child), "is_repository": (child / ".git").exists()})
            if len(entries) >= 300:
                break
        return {"roots": [str(item) for item in WORKSPACE_ROOTS], "current_path": str(current),
                "parent_path": str(current.parent) if current != root else None,
                "entries": entries, "search": False}

    def get(self, repo_id: str) -> Path:
        path = self._repositories.get(repo_id)
        if path is None:
            raise HTTPException(404, "Repository session expired. Validate the repository again.")
        return path

    def resolve_file(self, repo_id: str, relative_path: str) -> Path:
        rel = Path(relative_path)
        if rel.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise HTTPException(400, "Choose a supported source file within the repository.")
        return self.resolve_evidence_file(repo_id, relative_path)

    def resolve_evidence_file(self, repo_id: str, relative_path: str) -> Path:
        repo = self.get(repo_id)
        rel = Path(relative_path)
        if rel.is_absolute() or ".." in rel.parts:
            raise HTTPException(400, "Choose a file within the repository.")
        try:
            path = (repo / rel).resolve(strict=True)
            path.relative_to(repo)
        except (OSError, RuntimeError, ValueError):
            raise HTTPException(404, "The requested file was not found inside this repository.")
        if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
            raise HTTPException(413, "The selected file is missing or exceeds the 250 KB limit.")
        return path

    def files(self, repo_id: str) -> list[dict]:
        repo = self.get(repo_id)
        tracked = run_git(repo, ["ls-files", "-z"], timeout=15)
        untracked = run_git(repo, ["ls-files", "--others", "--exclude-standard", "-z"], timeout=15)
        result = []
        for raw in dict.fromkeys((tracked + "\0" + untracked).split("\0")):
            if not raw:
                continue
            rel = Path(raw)
            if rel.suffix.lower() not in SUPPORTED_EXTENSIONS or any(part in EXCLUDED_PARTS for part in rel.parts):
                continue
            try:
                path = self.resolve_file(repo_id, raw)
            except HTTPException:
                continue
            result.append({"path": raw.replace("\\", "/"), "size": path.stat().st_size})
            if len(result) >= 1500:
                break
        return result

    def context_files(self, repo_id: str) -> list[dict]:
        """List bounded source, test, and documentation files for investigation context."""
        repo = self.get(repo_id)
        tracked = run_git(repo, ["ls-files", "-z"], timeout=15)
        untracked = run_git(repo, ["ls-files", "--others", "--exclude-standard", "-z"], timeout=15)
        allowed = SUPPORTED_EXTENSIONS | {".md", ".rst", ".txt", ".toml", ".ini", ".cfg"}
        result = []
        for raw in dict.fromkeys((tracked + "\0" + untracked).split("\0")):
            if not raw:
                continue
            rel = Path(raw)
            if rel.suffix.lower() not in allowed or any(part in EXCLUDED_PARTS for part in rel.parts):
                continue
            try:
                path = self.resolve_evidence_file(repo_id, raw)
            except HTTPException as exc:
                if exc.status_code != 413:
                    continue
                try:
                    path = (repo / rel).resolve(strict=True)
                    path.relative_to(repo)
                    if not path.is_file():
                        continue
                except (OSError, RuntimeError, ValueError):
                    continue
            result.append({"path": raw.replace("\\", "/"), "size": path.stat().st_size})
            if len(result) >= 1500:
                break
        return result


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def run_git(repo: Path, args: list[str], *, timeout: int = 12, check: bool = True) -> str:
    try:
        result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "Git took too long to inspect this repository.")
    except OSError:
        raise HTTPException(500, "Git could not be started. Install Git and try again.")
    if check and result.returncode:
        raise HTTPException(400, "Git could not read that repository data.")
    return result.stdout[:300_000]
