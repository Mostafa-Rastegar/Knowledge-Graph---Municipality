"""One-off: screenshot the running frontend for the report."""
import sys
from playwright.sync_api import sync_playwright

OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/graph_output.png"
URL = "http://127.0.0.1:8000"

with sync_playwright() as p:
    for channel in ("msedge", "chrome"):
        try:
            browser = p.chromium.launch(channel=channel)
            break
        except Exception:
            browser = None
    if browser is None:
        browser = p.chromium.launch()  # bundled, if present
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto(URL, wait_until="networkidle")
    page.wait_for_timeout(3500)  # let vis-network physics settle
    page.screenshot(path=OUT, full_page=False)
    browser.close()
    print("saved", OUT)
