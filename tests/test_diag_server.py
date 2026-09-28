"""Diag dashboard: page, favicon, static assets and the cheap read-only endpoints.

Hardware- and model-backed routes (/api/record, /api/speak, /api/ask) are not
exercised here; they need the Jetson.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from oracle.diag import server  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    # No lifespan: tegrastats/ring-buffer wiring is not what's under test.
    return TestClient(server.app)


def test_index_serves_page_with_favicon_links(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    body = r.text
    assert "<title>Radio Oracle :: Diagnostics</title>" in body
    assert 'href="/favicon.ico"' in body
    assert 'href="/static/favicon.svg"' in body
    assert "fonts.googleapis.com" not in body, "fonts must be self-hosted (offline box)"


def test_favicon_ico_has_three_sizes(client: TestClient) -> None:
    r = client.get("/favicon.ico")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/x-icon"
    assert "max-age" in r.headers["cache-control"]
    assert r.content[:4] == b"\x00\x00\x01\x00"  # ICO header
    assert int.from_bytes(r.content[4:6], "little") == 3


@pytest.mark.parametrize(
    "name,ctype",
    [
        ("favicon.svg", "image/svg+xml"),
        ("apple-touch-icon.png", "image/png"),
        ("vt323-latin.woff2", "font/woff2"),
        ("share-tech-mono-latin.woff2", "font/woff2"),
    ],
)
def test_static_assets(client: TestClient, name: str, ctype: str) -> None:
    r = client.get(f"/static/{name}")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith(ctype)


@pytest.mark.parametrize("name", ["../server.py", "index.html", "nope.woff2", "server.py"])
def test_static_rejects_traversal_and_unknown(client: TestClient, name: str) -> None:
    r = client.get(f"/static/{name}")
    assert r.status_code == 404


def test_stats_reports_real_uptime_and_hostname(client: TestClient) -> None:
    j = client.get("/api/stats").json()
    assert 0 < j["uptime_sec"] < 10 * 365 * 86400  # was boot_time epoch before
    assert j["hostname"]
    assert j["memory"]["total_mb"] > 0


def test_procs_lists_every_unit_even_when_absent(client: TestClient) -> None:
    j = client.get("/api/procs").json()
    units = {s["unit"] for s in j["services"]}
    assert {"llama-server", "radio-oracle", "radio-oracle-tts", "radio-oracle-diag"} <= units
    for s in j["services"]:
        assert "present" in s
        if s["present"]:
            assert s["resident_mb"] >= 0 and s["swap_mb"] >= 0
    assert j["total_mb"] >= j["available_mb"] >= 0


def test_cgroup_memory_parses_v2_files(tmp_path, monkeypatch) -> None:
    cg = tmp_path / "radio-oracle.service"
    cg.mkdir()
    (cg / "memory.current").write_text("2097152\n")
    (cg / "memory.swap.current").write_text("1048576\n")
    (cg / "memory.stat").write_text("anon 1572864\nfile 524288\nkernel 0\n")
    monkeypatch.setattr(server, "_CGROUP_ROOT", tmp_path)
    out = server._cgroup_memory("radio-oracle")
    assert out == {
        "unit": "radio-oracle",
        "present": True,
        "resident_mb": 2.0,
        "anon_mb": 1.5,
        "file_mb": 0.5,
        "swap_mb": 1.0,
    }
    assert server._cgroup_memory("missing") == {"unit": "missing", "present": False}
