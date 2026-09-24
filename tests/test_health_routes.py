from __future__ import annotations

import json
from types import SimpleNamespace

from face_api.rest.routers.health import healthz, readyz


class _Store:
    def __init__(self, subjects=None, error: Exception | None = None):
        self.subjects = list(subjects or [])
        self.error = error

    def list_subjects(self):
        if self.error is not None:
            raise self.error
        return self.subjects


class _Recognition:
    def __init__(self, description=None, error: Exception | None = None):
        self.description = description or {"runtime": "onnx"}
        self.error = error

    def describe(self):
        if self.error is not None:
            raise self.error
        return self.description


def _request(store, recognition):
    state = SimpleNamespace(store=store, recognition_service=recognition)
    return SimpleNamespace(app=SimpleNamespace(state=state))


def test_healthz_reports_process_liveness():
    assert healthz() == {"code": 0, "msg": "success", "data": {"status": "ok"}}


def test_readyz_reports_ready_with_subject_count():
    response = readyz(_request(_Store([object(), object()]), _Recognition()))

    assert isinstance(response, dict)
    assert response["data"]["ready"] is True
    assert response["data"]["subjects"] == 2


def test_readyz_returns_503_when_a_required_component_fails():
    response = readyz(_request(_Store(), _Recognition(error=RuntimeError("model unavailable"))))

    assert response.status_code == 503
    payload = json.loads(response.body)
    assert payload["data"]["ready"] is False
    assert payload["data"]["runtime"] == {"error": "model unavailable"}
