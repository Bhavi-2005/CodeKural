# CodeKural

An evidence-backed code-history investigator. The Angular client browses tracked and untracked working-tree source files through a FastAPI backend, which collects real commits, diffs, blame, nearby tests and docs before calling Sarvam. New files without history remain browsable; explanations report that historical evidence is missing.

## Run locally

1. Copy `.env.example` to `.env`. Set `WORKSPACE_ROOT` to a directory containing repositories you want to inspect (leave blank to allow repositories under this project directory). Set `SARVAM_API_KEY` to enable explanations and test suggestions; repository browsing still works without it. The key is read only by the backend.
2. Start the API:

   ```powershell
   cd backend
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   python -m app.main
   ```

3. In another terminal, install and run the Angular app:

   ```powershell
   npm install
   npm start
   ```

Open `http://localhost:4200`. Enter an absolute path to a Git repository beneath `WORKSPACE_ROOT`, select a file and line range, and ask a question. The first configured root is the project directory when `WORKSPACE_ROOT` is omitted.

## Verify Sarvam chat

With `SARVAM_API_KEY` set in the project `.env`, install the official Python SDK and make one short chat request:

```powershell
python -m pip install sarvamai
python backend/first_sarvam_call.py
```

The script reads the key from `.env` and calls `sarvam-105b-conversations` through the SDK's V1 chat completions endpoint. It does not print the key.

For backend tests, install `backend/requirements-dev.txt`, then run `python -m pytest tests` from `backend`.

Selected source and retrieved evidence are sent to Sarvam only when an AI action is requested. Use public or non-sensitive repositories for this prototype.

## API

- `GET /api/health`
- `POST /api/repositories/validate`
- `GET /api/repositories/files?repository_id=...`
- `GET /api/repositories/history?repository_id=...&file_path=...&start_line=...&end_line=...`
- `POST /api/investigations`
- `POST /api/investigations/test-suggestion`

Repository identifiers are held in memory by the API process. Restarting the backend requires validating the repository again.
