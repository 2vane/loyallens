#!/usr/bin/env python
"""Render reports/slides.md -> reports/slides.pdf as a landscape 16:9 slide deck.

  python scripts/make_slides.py --in reports/slides.md --out reports/slides.pdf

Slides are separated by a line containing only '---'. The first slide is styled
as a title slide. Local images are inlined as base64 so the PDF is self-contained
(same trick as make_pdf.py). One slide per page.
"""
import argparse
import base64
import os
import re
import subprocess
import tempfile

import markdown

CSS = """
@page { size: 330mm 185mm; margin: 0; }        /* ~16:9 landscape */
* { box-sizing: border-box; }
body { font-family: "Helvetica Neue", Arial, sans-serif; color: #14213d; margin: 0; }
section.slide {
    position: relative; width: 330mm; height: 185mm; padding: 15mm 20mm 16mm;
    page-break-after: always; overflow: hidden;
    border-top: 8mm solid #14213d;
}
section.slide:last-child { page-break-after: auto; }
h1 { font-size: 40pt; margin: 0 0 6pt; color: #14213d; line-height: 1.05; }
h2 { font-size: 23pt; margin: 0 0 9pt; color: #1d4e89;
     border-bottom: 2px solid #e0a500; padding-bottom: 5pt; line-height: 1.15; }
h3 { font-size: 17pt; margin: 4pt 0; color: #333; font-weight: 500; }
p, li { font-size: 15pt; line-height: 1.4; color: #222; }
li { margin: 3pt 0; }
strong { color: #14213d; }
em { color: #1d4e89; }
blockquote { border-left: 4px solid #e0a500; margin: 8pt 0; padding: 2pt 14pt;
             font-style: italic; color: #444; font-size: 15pt; }
img { display: block; margin: 4pt auto; max-width: 68%; max-height: 82mm; }
table { border-collapse: collapse; width: 100%; font-size: 12pt; margin: 6pt 0; }
th, td { border: 1px solid #ccc; padding: 4pt 7pt; text-align: left; }
th { background: #14213d; color: #fff; }
tr:nth-child(even) td { background: #f5f6fa; }
.pageno { position: absolute; bottom: 6mm; right: 20mm; font-size: 10pt; color: #999; }
.brand  { position: absolute; bottom: 6mm; left: 20mm; font-size: 10pt; color: #999; }
/* Title slide */
section.title { border-top: 8mm solid #e0a500; padding-top: 42mm; }
section.title h1 { font-size: 60pt; color: #14213d; }
section.title h2 { font-size: 26pt; border: none; color: #1d4e89; }
section.title p { font-size: 17pt; }
"""


def inline_images(html, base_dir):
    def _sub(m):
        src = m.group(1)
        p = src if os.path.isabs(src) else os.path.normpath(os.path.join(base_dir, src))
        if os.path.exists(p):
            b64 = base64.b64encode(open(p, "rb").read()).decode()
            ext = os.path.splitext(p)[1].lstrip(".") or "png"
            return f'src="data:image/{ext};base64,{b64}"'
        return m.group(0)
    return re.sub(r'src="([^"]+\.(?:png|jpe?g|gif))"', _sub, html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="reports/slides.md")
    ap.add_argument("--out", dest="out", default="reports/slides.pdf")
    args = ap.parse_args()

    text = open(args.src, encoding="utf-8").read()
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    chunks = re.split(r"(?m)^---\s*$", text)
    base_dir = os.path.dirname(args.src) or "."

    slides = []
    n = len([c for c in chunks if c.strip()])
    idx = 0
    for chunk in chunks:
        if not chunk.strip():
            continue
        idx += 1
        body = markdown.markdown(chunk.strip(),
                                 extensions=["tables", "fenced_code", "sane_lists"])
        body = inline_images(body, base_dir)
        cls = "slide title" if idx == 1 else "slide"
        slides.append(
            f'<section class="{cls}">{body}'
            f'<div class="brand">LoyalLens · Team 2vane</div>'
            f'<div class="pageno">{idx} / {n}</div></section>')

    html = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<style>{CSS}</style></head><body>{''.join(slides)}</body></html>")
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as f:
        f.write(html)
        tmp = f.name
    try:
        subprocess.run(["weasyprint", tmp, args.out], check=True)
    finally:
        os.unlink(tmp)
    print(f"wrote {args.out} ({n} slides)")


if __name__ == "__main__":
    main()
