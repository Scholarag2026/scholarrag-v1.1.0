"""Verify open-access PDF availability for candidate seed papers.

Usage::

    python demo/tools/verify_pdf.py candidates.json verified.json [--browser-ua]

For every candidate the script

1. asks Unpaywall for ``best_oa_location.url_for_pdf`` (route ``unpaywall``); when
   Unpaywall has no PDF URL it falls back to the single OpenAlex
   ``best_oa_location.pdf_url`` carried in the candidate record (route ``openalex``);
   no other ``oa_locations`` are tried;
2. downloads that URL with a client equivalent to the backend's
   ``fetch_pdf_from_url`` (``httpx.Client(timeout=30.0, follow_redirects=True)``,
   httpx's default User-Agent, no extra headers) and accepts the paper only when the
   response is HTTP 200, ``content-type: application/pdf`` and the body starts with
   ``%PDF``.

``--browser-ua`` repeats step 2 with a Chrome-like User-Agent; it is kept only to
document hosts that block the default client and is *not* what the product does.

The Unpaywall contact address is read from ``UNPAYWALL_EMAIL`` (environment, then the
repository ``.env``). Nothing else is sent to any service.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
from collections import Counter
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
)
TODAY = datetime.date.today().isoformat()


def unpaywall_email() -> str:
    """UNPAYWALL_EMAIL from the environment, else from the repository ``.env``."""
    value = os.environ.get("UNPAYWALL_EMAIL", "").strip()
    if value:
        return value
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("UNPAYWALL_EMAIL="):
                value = line.split("=", 1)[1].split("#", 1)[0].strip().strip("'\"")
                if value:
                    return value
    sys.exit(
        "UNPAYWALL_EMAIL is not set. Export it or add UNPAYWALL_EMAIL=<your address> to "
        "the repository .env (Unpaywall requires a contact address)."
    )


def check_pdf(client: httpx.Client, url: str, *, browser_ua: bool = False) -> dict:
    headers = {"User-Agent": BROWSER_UA, "Accept": "application/pdf,*/*"} if browser_ua else None
    try:
        r = client.get(url, headers=headers)
    except Exception as exc:  # noqa: BLE001 - any transport error is a failed fetch
        return {"status": 0, "ctype": f"error:{type(exc).__name__}", "is_pdf": False,
                "final": url, "bytes": 0}
    ctype = r.headers.get("content-type", "").split(";")[0].strip()
    return {"status": r.status_code, "ctype": ctype, "is_pdf": r.content.startswith(b"%PDF"),
            "final": str(r.url), "bytes": len(r.content)}


def unpaywall(client: httpx.Client, doi: str, email: str) -> dict | None:
    r = client.get(f"https://api.unpaywall.org/v2/{doi}", params={"email": email})
    if r.status_code != 200:
        return None
    j = r.json()
    loc = j.get("best_oa_location") or {}
    return {"is_oa": j.get("is_oa"), "pdf_url": loc.get("url_for_pdf"), "url": loc.get("url"),
            "version": loc.get("version"), "license": loc.get("license"),
            "host_type": loc.get("host_type")}


def passes(res: dict) -> bool:
    return res["status"] == 200 and res["ctype"] == "application/pdf" and res["is_pdf"]


def verify(cands: list[dict], *, browser_ua: bool = False, log=print) -> list[dict]:
    email = unpaywall_email()
    # Same construction as backend/app/services/fulltext.py::fetch_pdf_from_url.
    client = httpx.Client(timeout=30.0, follow_redirects=True)
    out: list[dict] = []
    for c in cands:
        up = unpaywall(client, c["doi"], email)
        route, url = None, None
        if up and up.get("is_oa") and up.get("pdf_url"):
            route, url = "unpaywall", up["pdf_url"]
        elif c.get("pdf"):
            route, url = "openalex", c["pdf"]
        empty = {"status": 0, "ctype": "", "is_pdf": False, "final": None, "bytes": 0}
        res = check_pdf(client, url) if url else empty
        rec = dict(c, unpaywall=up, route=route, checked_url=url, verified_at=TODAY,
                   client="backend", http=res, ok=passes(res))
        if browser_ua and url:
            rec["http_browser_ua"] = check_pdf(client, url, browser_ua=True)
            rec["ok_browser_ua"] = passes(rec["http_browser_ua"])
        out.append(rec)
        flag = "OK  " if rec["ok"] else "FAIL"
        log(f"{flag} {c['doi']} {route} {res['status']} {res['ctype']}")
    return out


def summary(records: list[dict]) -> str:
    ok = [r for r in records if r["ok"]]
    routes = Counter(r["route"] for r in ok)
    journals = Counter(r["journal"] for r in ok)
    lines = [
        f"candidates: {len(records)}",
        f"passed (backend client): {len(ok)}",
        f"rejected: {len(records) - len(ok)}",
        f"routes among passed: {dict(routes)}",
        f"journals among passed: {dict(journals)}",
    ]
    if any("ok_browser_ua" in r for r in records):
        only_browser = [r["doi"] for r in records if r.get("ok_browser_ua") and not r["ok"]]
        lines.append(f"pass only with a browser User-Agent: {only_browser}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    if len(args) != 2:
        print(__doc__)
        return 2
    cands = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    records = verify(cands, browser_ua="--browser-ua" in argv)
    Path(args[1]).write_text(json.dumps(records, indent=1, ensure_ascii=False), encoding="utf-8")
    print(summary(records))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
