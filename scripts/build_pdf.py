"""Render the published report fragments to print-ready PDFs.

The artifact files are body fragments -- the publisher supplies the document
shell -- so each one is wrapped here, pinned to the light theme, and given a
print stylesheet before headless Chrome renders it. The charts are drawn by
script on load, so the render waits on a virtual time budget rather than
printing an empty frame.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

PRINT_CSS = """
<style>
  @page { size: A4; margin: 14mm 13mm 16mm; }
  html, body {
    background: #ffffff !important;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
  .page { max-width: none !important; padding: 0 !important; gap: 26px !important; }

  /* Keep a block and the heading that introduces it on one page. */
  h1, h2, h3 { break-after: avoid-page; }
  figure, table, .col, .surface, .keybox, .verdict, .status-row, .split {
    break-inside: avoid-page;
  }
  section { break-inside: auto; }

  /* Tables scroll on screen; on paper they have to fit the measure. */
  .scroller { overflow: visible !important; }
  table { min-width: 0 !important; font-size: 11.5px !important; }
  th, td { padding: 6px 8px !important; }

  /* The reader cannot open a disclosure widget on paper. */
  details.table-view { display: none !important; }

  body { font-size: 11.5pt; line-height: 1.55; }
  .measure { max-width: none !important; }
  h1 { font-size: 26pt !important; }
  h2 { font-size: 15pt !important; }
  .standfirst { font-size: 12pt !important; }
  a { color: inherit; text-decoration: none; }
</style>
"""


def find_chrome() -> str:
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    found = shutil.which("chrome") or shutil.which("msedge")
    if found:
        return found
    raise FileNotFoundError("No Chrome or Edge binary found for PDF rendering")


def wrap(fragment: str) -> str:
    """Give the fragment a document shell pinned to the light theme."""
    return (
        '<!doctype html><html lang="tr" data-theme="light"><head>'
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<style>body{margin:0;font:14px system-ui;}img{max-width:100%}"
        "[hidden]{display:none!important}</style>"
        f"{fragment}{PRINT_CSS}"
        "</head><body></body></html>"
    )


def render(source: Path, target: Path, chrome: str, *, budget_ms: int = 9000) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    # Chrome resolves --print-to-pdf against its own working directory, not ours.
    target = target.resolve()
    document = wrap(source.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as workspace:
        staged = Path(workspace) / "page.html"
        staged.write_text(document, encoding="utf-8")
        profile = Path(workspace) / "profile"
        command = [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            f"--user-data-dir={profile}",
            f"--virtual-time-budget={budget_ms}",
            "--run-all-compositor-stages-before-draw",
            "--no-pdf-header-footer",
            f"--print-to-pdf={target}",
            staged.resolve().as_uri(),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        if not target.exists():
            raise RuntimeError(
                f"Chrome produced no PDF for {source.name}\n"
                f"stdout: {result.stdout[-800:]}\nstderr: {result.stderr[-800:]}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs", nargs="+", help="source.html:target.pdf")
    args = parser.parse_args()
    chrome = find_chrome()
    print(f"renderer: {chrome}")
    for pair in args.pairs:
        source, _, target = pair.rpartition(":")
        render(Path(source), Path(target), chrome)
        size = Path(target).stat().st_size
        print(f"  {Path(target).name}  {size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
