"""Render a Persian Markdown report to PDF.

The project already installs Playwright for the web interface, so we print the
page with Chromium instead of adding a new tool. Mermaid blocks render first,
then the page becomes a PDF.

Run: python -m src.md2pdf docs/FINAL_REPORT.md --out docs/FINAL_REPORT.pdf
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

# The project keeps its browser in ms-playwright/. Point Playwright there before
# it loads, so the command works without an extra environment variable.
_LOCAL_BROWSERS = Path(__file__).resolve().parent.parent / "ms-playwright"
if _LOCAL_BROWSERS.is_dir():
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(_LOCAL_BROWSERS))

import markdown  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: "Vazirmatn", "Segoe UI", Tahoma, sans-serif; direction: rtl;
       text-align: right; line-height: 1.9; font-size: 11pt; color: #111; }
h1 { font-size: 19pt; border-bottom: 3px solid #1f4e79; padding-bottom: 6px; color: #1f4e79; }
h2 { font-size: 15pt; margin-top: 26px; color: #1f4e79; border-bottom: 1px solid #ccd; padding-bottom: 3px; }
h3 { font-size: 12.5pt; margin-top: 18px; color: #244; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 9.5pt; }
th, td { border: 1px solid #bbb; padding: 5px 7px; }
th { background: #eef2f7; font-weight: 600; }
tr:nth-child(even) td { background: #fafbfc; }
code, pre { direction: ltr; text-align: left; font-family: Consolas, monospace; font-size: 9pt; }
pre { background: #f5f6f8; border: 1px solid #ddd; border-radius: 4px; padding: 9px; overflow-x: auto; }
img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
blockquote { border-right: 4px solid #1f4e79; margin: 0; padding-right: 12px; color: #444; }
.mermaid { direction: ltr; text-align: center; background: #fff; padding: 8px; }
h2, h3 { page-break-after: avoid; }
table, pre, .mermaid { page-break-inside: avoid; }
"""

PAGE = """<!doctype html><html lang="fa" dir="rtl"><head><meta charset="utf-8">
<link href="https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<style>{css}</style></head><body>{body}
<script>
  if (window.mermaid) {{ mermaid.initialize({{startOnLoad: true, theme: "neutral"}}); }}
</script>
</body></html>"""


def to_html(md_text: str) -> str:
    """Convert Markdown to HTML and keep mermaid blocks for the browser."""
    fences = []

    def stash(match: re.Match) -> str:
        fences.append(match.group(1))
        return f"@@MERMAID{len(fences) - 1}@@"

    md_text = re.sub(r"```mermaid\n(.*?)```", stash, md_text, flags=re.S)
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "toc"])
    for i, code in enumerate(fences):
        body = body.replace(f"@@MERMAID{i}@@", f'<div class="mermaid">{code}</div>')
    return PAGE.format(css=CSS, body=body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a Markdown report to PDF")
    parser.add_argument("source")
    parser.add_argument("--out")
    args = parser.parse_args()

    src = Path(args.source)
    out = Path(args.out or src.with_suffix(".pdf"))
    html_path = src.with_suffix(".render.html")
    html_path.write_text(to_html(src.read_text(encoding="utf-8")), encoding="utf-8")

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
        page.wait_for_timeout(2500)  # let mermaid draw
        page.pdf(path=str(out), format="A4", print_background=True,
                 margin={"top": "18mm", "bottom": "18mm", "left": "16mm", "right": "16mm"})
        browser.close()
    html_path.unlink()
    print(f"{out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
