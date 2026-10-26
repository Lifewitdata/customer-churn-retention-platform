"""Capture every dashboard page to docs/screenshots/ with headless Chrome.

Streamlit renders over a websocket, so a plain `chrome --screenshot` fires before the
data arrives and writes a blank page. This drives Chrome DevTools Protocol instead:
launch Chrome headless, set a desktop viewport, navigate, poll until every block has
streamed in (no skeleton or spinner left, all images decoded, page text no longer
growing), then capture beyond the fold.

    python -m streamlit run dashboard/app.py --server.port 8531 --server.headless true
    python dashboard/capture_screenshots.py

Needs Google Chrome (or Microsoft Edge) installed. Python deps are the ones the API
stack already brings in: `requests` and `websockets`.
"""
from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests
from websockets.sync.client import connect

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "docs" / "screenshots"
APP = "http://localhost:8531"
PORT = 9333
WIDTH, HEIGHT, SCALE = 1440, 1000, 2
# heading text per page id - the "the app finished rendering" signal
PAGES = {
    "overview": "Executive overview",
    "segments": "Risk segmentation",
    "drivers": "Driver analysis",
    "survival": "Survival and timing",
    "fairness": "Fairness and subpopulation performance",
    "actions": "Action list",
    "health": "Model health and drift",
}
BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]

BODY_TEXT = "document.body ? document.body.innerText : ''"
RENDER_STATE = (
    "(() => { const s = document.querySelectorAll("
    "'[data-testid=\"stSkeleton\"],[data-testid=\"stSpinner\"],[data-testid=\"stStatusWidget\"]').length;"
    " const i = Array.from(document.images).filter(im => !im.complete || im.naturalWidth === 0).length;"
    " return s + '|' + i + '|' + (document.body ? document.body.innerText.length : 0); })()")
SCROLL_TOP = ("window.scrollTo(0, 0); document.querySelectorAll('main,[data-testid=\"stMain\"],"
              "'[data-testid=\"stAppViewContainer\"]').forEach(el => {el.scrollTo(0, 0); "
              "el.scrollLeft = 0;}); 'ok'")
# Streamlit scrolls inside <main>, so the document height reads 0 - measure the scroll container.
CONTENT_HEIGHT = ("(() => {const m = document.querySelector('[data-testid=\"stMain\"]');"
                  " return Math.ceil(Math.max(m ? m.scrollHeight : 0, document.body.scrollHeight));})()")


def find_browser() -> str:
    for p in BROWSERS:
        if Path(p).exists():
            return p
    sys.exit("no Chrome or Edge found - add the path to BROWSERS in this script")


def start_chrome(profile: Path) -> subprocess.Popen:
    proc = subprocess.Popen(
        [find_browser(), "--headless=new", "--disable-gpu", "--hide-scrollbars",
         "--disable-extensions", "--no-first-run", f"--remote-debugging-port={PORT}",
         f"--user-data-dir={profile}", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            if requests.get(f"http://127.0.0.1:{PORT}/json/version", timeout=2).ok:
                return proc
        except requests.RequestException:
            pass
        time.sleep(0.5)
    proc.terminate()
    sys.exit(f"Chrome did not open a DevTools port on {PORT}")


class Tab:
    """One CDP websocket with the request/reply plumbing kept to a dozen lines."""

    def __init__(self, ws_url: str):
        self._ws_url = ws_url
        self._next_id = 0

    def __enter__(self) -> "Tab":
        self._cm = connect(self._ws_url, max_size=None)
        self.ws = self._cm.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        self._cm.__exit__(*exc)

    def call(self, method: str, **params) -> dict:
        self._next_id += 1
        mine = self._next_id
        self.ws.send(json.dumps({"id": mine, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") != mine:
                continue  # skip CDP events
            if "error" in msg:
                raise RuntimeError(f"{method}: {msg['error']['message']}")
            return msg.get("result", {})

    def eval_(self, expression: str) -> str:
        return self.call("Runtime.evaluate", expression=expression, returnByValue=True)["result"].get("value", "")

    def wait_for(self, text: str, timeout: float = 90.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if text in self.eval_(BODY_TEXT):
                return
            time.sleep(0.5)
        raise RuntimeError(f"a page never rendered {text!r} - is the dashboard up at {APP}?")

    def wait_until_settled(self, timeout: float = 120.0, stable_for: int = 4) -> None:
        """Streamlit streams blocks while the script runs, so 'heading is present' is far too early.

        A page is ready once no skeleton/spinner node is left, every <img> has decoded and the
        visible text length has stopped changing - then a short beat for the SVG marks to paint.
        """
        deadline = time.time() + timeout
        last, still = None, 0
        while time.time() < deadline:
            skeletons, images, length = self.eval_(RENDER_STATE).split("|")
            if skeletons == "0" and images == "0" and length == last:
                still += 1
            else:
                still = 0
                last = length
            if still >= stable_for:
                time.sleep(1.5)
                return
            time.sleep(0.4)
        raise RuntimeError(f"dashboard never settled - {self.eval_(RENDER_STATE)}")


def main() -> int:
    try:
        requests.get(f"{APP}/_stcore/health", timeout=3)
    except requests.RequestException:
        sys.exit(f"dashboard not reachable at {APP} - start it with: streamlit run dashboard/app.py")

    profile = Path(tempfile.mkdtemp(prefix="chrome-capture-"))
    chrome = start_chrome(profile)
    try:
        new = requests.put(f"http://127.0.0.1:{PORT}/json/new?about:blank", timeout=5)
        new.raise_for_status()
        OUT.mkdir(parents=True, exist_ok=True)
        with Tab(new.json()["webSocketDebuggerUrl"]) as tab:
            tab.call("Page.enable")
            tab.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=HEIGHT,
                     deviceScaleFactor=SCALE, mobile=False)
            for page_id, heading in PAGES.items():
                tab.call("Page.navigate", url=f"{APP}/?page={page_id}")
                tab.wait_for(heading)
                tab.wait_until_settled()
                # captureBeyondViewport only returns the viewport here (Streamlit scrolls inside
                # <main>) and slides the fixed sidebar off-canvas, so grow the viewport instead.
                full = int(tab.eval_(CONTENT_HEIGHT))
                tab.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=max(full, HEIGHT),
                         deviceScaleFactor=SCALE, mobile=False)
                tab.wait_until_settled()
                tab.call("Runtime.evaluate", expression=SCROLL_TOP)
                shot = tab.call("Page.captureScreenshot", format="png")
                path = OUT / f"dashboard_{page_id}.png"
                path.write_bytes(base64.b64decode(shot["data"]))
                print(f"  {path.name}: {path.stat().st_size / 1024:.0f} KB, {WIDTH}x{full} logical")
                tab.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=HEIGHT,
                         deviceScaleFactor=SCALE, mobile=False)
    finally:
        chrome.terminate()
        chrome.wait(timeout=15)
        shutil.rmtree(profile, ignore_errors=True)
    print(f"wrote {len(PAGES)} screenshots to {OUT.relative_to(BASE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
