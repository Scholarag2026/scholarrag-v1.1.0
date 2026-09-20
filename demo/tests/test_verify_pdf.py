"""Offline tests for demo/tools/verify_pdf.py (client construction, pass rule, e-mail)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import httpx
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load_module():
    spec = importlib.util.spec_from_file_location("verify_pdf", TOOLS / "verify_pdf.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_pdf"] = module
    spec.loader.exec_module(module)
    return module


def test_email_comes_from_env_or_repo_dotenv_and_never_from_source(monkeypatch, tmp_path):
    vp = _load_module()
    source = (TOOLS / "verify_pdf.py").read_text(encoding="utf-8")
    assert "@" not in source.replace("<your address>", "")  # no literal address shipped
    monkeypatch.setenv("UNPAYWALL_EMAIL", "reviewer@example.org")
    assert vp.unpaywall_email() == "reviewer@example.org"
    monkeypatch.delenv("UNPAYWALL_EMAIL")
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "FOO=1\nUNPAYWALL_EMAIL=from-dotenv@example.org  # comment\n", encoding="utf-8"
    )
    monkeypatch.setattr(vp, "REPO_ROOT", tmp_path)
    assert vp.unpaywall_email() == "from-dotenv@example.org"
    dotenv.write_text("UNPAYWALL_EMAIL=\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="UNPAYWALL_EMAIL is not set"):
        vp.unpaywall_email()


def test_pass_rule_requires_200_pdf_content_type_and_magic():
    vp = _load_module()
    assert vp.passes({"status": 200, "ctype": "application/pdf", "is_pdf": True})
    assert not vp.passes({"status": 200, "ctype": "text/html", "is_pdf": False})
    assert not vp.passes({"status": 200, "ctype": "application/pdf", "is_pdf": False})
    assert not vp.passes({"status": 403, "ctype": "application/pdf", "is_pdf": True})


def test_check_pdf_uses_httpx_default_user_agent_unless_browser_mode():
    vp = _load_module()
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("user-agent", ""))
        return httpx.Response(
            200, content=b"%PDF-1.7 ...", headers={"content-type": "application/pdf"}
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler), timeout=30.0, follow_redirects=True
    )
    res = vp.check_pdf(client, "https://host.example/paper.pdf")
    assert res == {"status": 200, "ctype": "application/pdf", "is_pdf": True,
                   "final": "https://host.example/paper.pdf", "bytes": 12}
    vp.check_pdf(client, "https://host.example/paper.pdf", browser_ua=True)
    assert seen[0].startswith("python-httpx/") and seen[1].startswith("Mozilla/5.0")
