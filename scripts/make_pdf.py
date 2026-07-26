#!/usr/bin/env python
"""Render reports/report.md -> reports/report.pdf in the sprint template style.

  python scripts/make_pdf.py --in reports/report.md --out reports/report.pdf

Strips HTML comments (the [[N]] fill-notes), converts Markdown (tables + fenced
code) to HTML, wraps it in print CSS (A4, numbered-feel headings, centered title
block), and calls the weasyprint CLI. Warns if any [[...]] placeholders remain.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

import markdown

CSS = """
@page { size: A4; margin: 2cm 2.2cm; }
body { font-family: "Times New Roman", Georgia, serif; font-size: 10.5pt;
       line-height: 1.42; color: #111; }
h1 { font-size: 17pt; text-align: center; margin: 0 0 2pt; line-height: 1.25; }
h1 + p { text-align: center; }            /* author / affiliation block */
h2 { font-size: 12.5pt; border-bottom: 1px solid #999; padding-bottom: 2pt;
     margin: 14pt 0 6pt; }
h3 { font-size: 11pt; margin: 9pt 0 3pt; }
p, li { text-align: justify; }
code, pre { font-family: "DejaVu Sans Mono", monospace; font-size: 9pt; }
pre { background: #f4f4f4; padding: 6pt; border-radius: 3px; white-space: pre-wrap; }
table { border-collapse: collapse; width: 100%; font-size: 9.5pt; margin: 6pt 0; }
th, td { border: 1px solid #bbb; padding: 3pt 5pt; text-align: left; }
em { color: #222; }
a { color: #0b3d91; text-decoration: none; word-break: break-all; }
.placeholder { background: #ffe08a; }      /* highlight any leftover [[...]] */
img { max-width: 100%; display: block; margin: 8pt auto; }
h1 em { font-size: inherit; font-style: italic; }   /* don't shrink title emphasis */
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="reports/report.md")
    ap.add_argument("--out", dest="out", default="reports/report.pdf")
    args = ap.parse_args()

    text = open(args.src, encoding="utf-8").read()
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)  # drop fill-notes

    leftover = re.findall(r"\[\[.*?\]\]", text)
    if leftover:
        print(f"WARNING: {len(leftover)} unfilled placeholder(s) remain: "
              f"{sorted(set(leftover))[:8]}{'...' if len(set(leftover)) > 8 else ''}",
              file=sys.stderr)

    html_body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])

    # Inline local images as base64 data URIs so the PDF is self-contained and
    # path-independent (weasyprint resolves relative paths against a temp dir).
    import base64
    def _inline(m):
        src = m.group(1)
        p = src if os.path.isabs(src) else os.path.normpath(
            os.path.join(os.path.dirname(args.src) or ".", src))  # relative to the .md's dir
        if os.path.exists(p):
            b64 = base64.b64encode(open(p, "rb").read()).decode()
            ext = os.path.splitext(p)[1].lstrip(".") or "png"
            return f'src="data:image/{ext};base64,{b64}"'
        return m.group(0)
    html_body = re.sub(r'src="([^"]+\.(?:png|jpe?g|gif))"', _inline, html_body)
    # highlight any surviving [[...]] so they are impossible to miss in the PDF
    html_body = re.sub(r"(\[\[.*?\]\])", r'<span class="placeholder">\1</span>', html_body)
    html = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style>" \
           f"</head><body>{html_body}</body></html>"

    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as f:
        f.write(html)
        tmp = f.name
    try:
        subprocess.run(["weasyprint", tmp, args.out], check=True)
    finally:
        os.unlink(tmp)
    print(f"wrote {args.out}" + ("  (with UNFILLED placeholders)" if leftover else ""))


if __name__ == "__main__":
    main()
