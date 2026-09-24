from __future__ import annotations

import importlib.util
import io
import json
import sqlite3
import sys
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "deploy" / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


preflight = _load_script("preflight.py")
healthcheck = _load_script("healthcheck.py")
backup = _load_script("backup.py")
sbom_generator = _load_script("generate_sbom.py")


def test_load_env_file_supports_comments_export_and_quotes(tmp_path):
    env_file = tmp_path / "face.env"
    env_file.write_text(
        "# comment\nexport FACE_API_RUNTIME=onnx\nFACE_API_PORT='8000'\n",
        encoding="utf-8",
    )

    assert preflight.load_env_file(env_file) == {
        "FACE_API_RUNTIME": "onnx",
        "FACE_API_PORT": "8000",
    }


def test_load_env_file_rejects_malformed_lines(tmp_path):
    env_file = tmp_path / "face.env"
    env_file.write_text("NOT_AN_ASSIGNMENT\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected KEY=VALUE"):
        preflight.load_env_file(env_file)


def test_onnx_preflight_checks_models_and_writable_directories(tmp_path):
    app_dir = tmp_path / "app"
    data_dir = tmp_path / "data"
    log_dir = tmp_path / "log"
    for directory in (app_dir, data_dir, log_dir):
        directory.mkdir()
    model_paths = [
        app_dir / "models/onnx/Retinaface_mobilenet0.25.onnx",
        app_dir / "models/onnx/cvlface_adaface_ir50_webface4m.onnx",
        app_dir / "models/onnx/cvlface_dfa_mobilenet.onnx",
    ]
    for model in model_paths:
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(b"model")

    args = Namespace(
        env_file=tmp_path / "missing.env",
        app_dir=app_dir,
        data_dir=data_dir,
        log_dir=log_dir,
        runtime="onnx",
        skip_runtime_import=True,
        deep_rknn_check=False,
    )
    checks = preflight.collect_checks(args, {"PATH": "/usr/bin:/bin"})

    assert checks
    assert all(check.ok for check in checks), [(check.name, check.detail) for check in checks]


def test_verify_rknn_manifest_detects_model_tampering(tmp_path):
    model = tmp_path / "detector.rknn"
    model.write_bytes(b"expected")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "rknn_path": model.name,
                        "rknn_sha256": preflight.sha256_file(model),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert all(check.ok for check in preflight.verify_rknn_manifest(manifest, [model]))
    model.write_bytes(b"tampered")
    assert not preflight.verify_rknn_manifest(manifest, [model])[0].ok


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_health_probe_requires_ready_boolean(monkeypatch):
    response = _Response(json.dumps({"code": 0, "data": {"ready": True}}).encode())
    monkeypatch.setattr(healthcheck.urllib.request, "urlopen", lambda *args, **kwargs: response)

    assert healthcheck.probe("http://localhost/readyz", 1) == (True, "ready")


def test_health_failure_counter_is_atomic(tmp_path):
    state = tmp_path / "health-failures"

    assert healthcheck.read_failures(state) == 0
    healthcheck.write_failures(state, 2)
    assert healthcheck.read_failures(state) == 2


def test_restart_rejects_untrusted_service_name():
    restarted, detail = healthcheck.restart_service("face-serve.service;reboot")

    assert restarted is False
    assert "refusing" in detail


def test_backup_uses_sqlite_backup_and_copies_subjects(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with sqlite3.connect(data_dir / "face_api.db") as connection:
        connection.execute("CREATE TABLE subjects(id TEXT)")
        connection.execute("INSERT INTO subjects VALUES ('one')")
    subjects = data_dir / "subjects"
    subjects.mkdir()
    (subjects / "one.jpg").write_bytes(b"image")

    destination = backup.backup_data(data_dir, tmp_path / "backups", "test")

    with sqlite3.connect(destination / "face_api.db") as connection:
        assert connection.execute("SELECT id FROM subjects").fetchone() == ("one",)
    assert (destination / "subjects/one.jpg").read_bytes() == b"image"


def test_systemd_service_uses_preflight_and_restart_limits():
    unit = (ROOT / "deploy/systemd/face-serve.service").read_text(encoding="utf-8")

    assert "ExecStartPre=" in unit
    assert "Restart=on-failure" in unit
    assert "StartLimitBurst=5" in unit
    assert "EnvironmentFile=-/etc/face-serve/face-serve.env" in unit


def test_package_manifest_requires_source_commit():
    schema = json.loads(
        (ROOT / "deploy/packaging/manifest.schema.json").read_text(encoding="utf-8")
    )

    assert "source_commit" in schema["required"]
    assert schema["properties"]["source_commit"]["minLength"] == 1


def test_archive_checksum_uses_a_portable_basename():
    script = (ROOT / "deploy/scripts/build-offline-package.sh").read_text(
        encoding="utf-8"
    )

    assert 'cd "${OUTPUT_DIR}"' in script
    assert 'sha256sum "${PACKAGE_NAME}.tar.zst"' in script


def test_healthcheck_default_url_uses_service_port(monkeypatch):
    monkeypatch.setenv("FACE_API_PORT", "8123")

    args = healthcheck.build_parser().parse_args([])

    assert args.url == "http://127.0.0.1:8123/readyz"


def test_health_supervisor_has_independent_persistent_runtime_state():
    unit = (ROOT / "deploy/systemd/face-serve-healthcheck.service").read_text(encoding="utf-8")

    assert "EnvironmentFile=-/etc/face-serve/face-serve.env" in unit
    assert "RuntimeDirectory=face-serve-health" in unit
    assert "RuntimeDirectoryPreserve=yes" in unit
    assert "/run/face-serve-health/health-failures" in unit
    assert "http://127.0.0.1:8000" not in unit


def test_health_supervisor_restarts_only_after_threshold(monkeypatch, tmp_path):
    state = tmp_path / "failures"
    restarts = []
    monkeypatch.setattr(healthcheck, "probe", lambda url, timeout: (False, "down"))
    monkeypatch.setattr(healthcheck, "service_is_active", lambda service: True)

    def _restart(service):
        restarts.append(service)
        return True, "restarted"

    monkeypatch.setattr(healthcheck, "restart_service", _restart)
    arguments = [
        "--state-file",
        str(state),
        "--failure-threshold",
        "2",
        "--restart-service",
        "face-serve.service",
        "--supervisor",
    ]

    assert healthcheck.main(arguments) == 0
    assert restarts == []
    assert healthcheck.read_failures(state) == 1

    assert healthcheck.main(arguments) == 0


def test_sbom_contains_wheel_and_model_checksums(tmp_path):
    package = tmp_path / "package"
    wheelhouse = package / "packages/wheelhouse"
    models = package / "models/onnx"
    wheelhouse.mkdir(parents=True)
    models.mkdir(parents=True)
    (package / "manifest.json").write_text(
        json.dumps(
            {
                "version": "1.2.3",
                "profile": "linux-x86_64-onnx-cpu",
                "created_at": "2026-09-23T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    wheel = wheelhouse / "example-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "example-1.0.dist-info/METADATA",
            "Name: example\nVersion: 1.0\nLicense: MIT\nSummary: fixture\n",
        )
    (models / "model.onnx").write_bytes(b"model")

    sbom_path, licenses_path = sbom_generator.generate(package, package / "sbom")
    sbom = json.loads(sbom_path.read_text(encoding="utf-8"))
    licenses = json.loads(licenses_path.read_text(encoding="utf-8"))

    assert sbom["spdxVersion"] == "SPDX-2.3"
    assert any(item["name"] == "example" for item in sbom["packages"])
    assert any(item["name"] == "models/onnx/model.onnx" for item in sbom["packages"])
    assert licenses[0]["license"] == "MIT"
