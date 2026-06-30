from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def topic11_request_payload(project_root: Path) -> dict[str, Any]:
    return json.loads((project_root / "examples" / "topic11_request.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def topic11_response_payload(project_root: Path) -> dict[str, Any]:
    return json.loads((project_root / "examples" / "topic11_response.json").read_text(encoding="utf-8"))
