import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "").strip()
SARVAM_MODEL = os.getenv("SARVAM_MODEL", "sarvam-105b-conversations").strip()
BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))

_root_setting = os.getenv("WORKSPACE_ROOT", str(PROJECT_ROOT)).strip()
_roots = (_root_setting or str(PROJECT_ROOT)).split(os.pathsep)
WORKSPACE_ROOTS = tuple(Path(root).expanduser().resolve() for root in _roots if root.strip())
