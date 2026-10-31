import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import BACKEND_HOST, BACKEND_PORT, SARVAM_API_KEY
from .git_service import file_history
from .investigation_service import investigate
from .repository_service import RepositoryStore
from .schemas import InvestigationRequest, TestSuggestionRequest, ValidateRepositoryRequest

app = FastAPI(title="CodeKural API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:4200", "http://127.0.0.1:4200"],
                   allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
repositories = RepositoryStore()


@app.get("/api/health")
def health():
    return {"status": "ok", "ai_configured": bool(SARVAM_API_KEY)}


@app.post("/api/repositories/validate")
def validate_repository(request: ValidateRepositoryRequest):
    return repositories.validate(request.path)


@app.get("/api/repositories/browse")
def browse_repositories(path: str | None = None, search: str = Query(default="", max_length=120)):
    return repositories.browse(path, search)


@app.post("/api/repositories/import-folder")
async def import_repository_folder(files: list[UploadFile] = File(...)):
    """Import a folder selected with the browser's native directory picker."""
    if not files:
        raise HTTPException(400, "Choose a folder that contains files.")
    if len(files) > 50_000:
        raise HTTPException(413, "That folder contains too many files to import.")

    temp_root = Path(tempfile.mkdtemp(prefix="codekural-import-"))
    total_bytes = 0
    imported = 0
    root_name = "Imported repository"
    try:
        for upload in files:
            raw_name = (upload.filename or "").replace("\\", "/")
            relative = PurePosixPath(raw_name)
            if relative.is_absolute() or not relative.parts or ".." in relative.parts:
                continue
            if len(relative.parts) > 1:
                root_name = relative.parts[0]
                relative = PurePosixPath(*relative.parts[1:])
            if not relative.parts or any(part in {"node_modules", ".venv", "venv", "dist", "build", "coverage", "__pycache__", ".next", "vendor"} for part in relative.parts):
                continue
            target = (temp_root / Path(*relative.parts)).resolve()
            try:
                target.relative_to(temp_root.resolve())
            except ValueError:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as destination:
                while chunk := await upload.read(1024 * 1024):
                    total_bytes += len(chunk)
                    if total_bytes > 300 * 1024 * 1024:
                        raise HTTPException(413, "The selected folder exceeds the 300 MB import limit.")
                    destination.write(chunk)
            imported += 1
            await upload.close()

        if imported == 0:
            raise HTTPException(400, "No files from that folder could be imported.")
        (temp_root / ".codekural-import-name").write_text(root_name, encoding="utf-8")
        if not (temp_root / ".git").exists():
            initialized = subprocess.run(["git", "-C", str(temp_root), "init"], capture_output=True,
                                         text=True, timeout=15, check=False)
            if initialized.returncode:
                raise HTTPException(500, "Git could not prepare the imported folder.")
        info = repositories.validate_selected(str(temp_root))
        info["name"] = root_name
        return info
    except HTTPException:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    except (OSError, subprocess.TimeoutExpired):
        shutil.rmtree(temp_root, ignore_errors=True)
        raise HTTPException(500, "CodeKural could not prepare that folder for local browsing.")


@app.get("/api/repositories/files")
def list_files(repository_id: str = Query(min_length=8, max_length=64)):
    return {"files": repositories.files(repository_id)}


@app.get("/api/repositories/history")
def history(repository_id: str, file_path: str, start_line: int | None = Query(default=None, ge=1),
            end_line: int | None = Query(default=None, ge=1)):
    if (start_line is None) != (end_line is None):
        raise HTTPException(400, "Provide both start_line and end_line.")
    if start_line is not None and end_line is not None and (end_line < start_line or end_line - start_line > 199):
        raise HTTPException(400, "Choose a valid history line range of up to 200 lines.")
    result = file_history(repositories, repository_id, file_path, start_line, end_line)
    return result


@app.get("/api/repositories/source")
def source(repository_id: str, file_path: str):
    path = repositories.resolve_file(repository_id, file_path)
    content = path.read_text(encoding="utf-8", errors="replace")
    if len(content) > 250_000:
        raise HTTPException(413, "The selected file exceeds the 250 KB limit.")
    return {"content": content, "line_count": max(1, len(content.splitlines()))}


@app.post("/api/investigations")
def create_investigation(request: InvestigationRequest):
    return investigate(repositories, request)


@app.post("/api/investigations/test-suggestion")
def suggest_test(request: TestSuggestionRequest):
    return investigate(repositories, request.investigation, test=True, framework=request.framework,
                       hypothesis=request.hypothesis)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=BACKEND_HOST, port=BACKEND_PORT)
