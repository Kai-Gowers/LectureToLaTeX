#!/usr/bin/env python3
import argparse, json, re, subprocess, sys, pathlib
from PIL import Image
import pytesseract
from jinja2 import Template

# --- 0) Config ---
TEX_TEMPLATE = r"""
\documentclass[11pt]{article}
\usepackage{amsmath,amsthm,amssymb,mathtools}
\usepackage[margin=1in]{geometry}
\usepackage{enumitem}
\newtheorem{definition}{Definition}[section]
\newtheorem{theorem}{Theorem}[section]
\newtheorem{lemma}{Lemma}[section]
\title{Lecture — Board Notes}
\date{\today}
\begin{document}\maketitle\tableofcontents

{{ body }}

\end{document}
"""

HEADING_RE = re.compile(r'^\s*(Definition|Theorem|Lemma|Proposition|Corollary|Remark|Example)\b[:\-\s]*(.*)$', re.I)
BULLET_RE  = re.compile(r'^\s*([\-•\*\u2022]|->)\s+(.*)$')
MATH_HINT  = re.compile(r'(=|≥|≤|→|∑|∫|∇|∂|[A-Za-z]\^\d|\\(?:frac|sum|int|prod|nabla|partial)|\b(?:lim|sin|cos|tan|log|ln)\b)')

REMAP = {
    '->': r'\to ', '⇒': r'\Rightarrow ', '≥': r'\ge ', '≤': r'\le ',
    '±': r'\pm ', '×': r'\times ', '·': r'\cdot ', '∞': r'\infty ',
    'α': r'\alpha ', 'β': r'\beta ', 'γ': r'\gamma ', 'δ': r'\delta ',
    'ε': r'\varepsilon ', 'λ': r'\lambda ', 'μ': r'\mu ', 'π': r'\pi ',
    'σ': r'\sigma ', 'θ': r'\theta ', 'Δ': r'\Delta ', '∑': r'\sum ', '∫': r'\int ',
}

def to_latex_safe(s: str) -> str:
    for k,v in REMAP.items(): s = s.replace(k,v)
    # naive superscript fix: x^2 -> x^{2}
    s = re.sub(r'\^([A-Za-z0-9])', r'^{\1}', s)
    return s.strip()

# --- 1) OCR to lines (with positions) ---
def ocr_lines_with_boxes(image_path: str):
    img = Image.open(image_path)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    lines = {}  # line_num -> {"text":..., "left":min, "top":min, "right":max, "bottom":max}
    for i in range(len(data['text'])):
        if int(data['conf'][i]) < 40:  # drop very low-confidence fragments
            continue
        text = data['text'][i].strip()
        if not text: continue
        ln = data['line_num'][i] + 1000*data['block_num'][i]  # unique per visual line
        left, top, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        if ln not in lines:
            lines[ln] = {"chunks": [], "left": left, "top": top, "right": left+w, "bottom": top+h}
        lines[ln]["chunks"].append(text)
        lines[ln]["left"]   = min(lines[ln]["left"], left)
        lines[ln]["top"]    = min(lines[ln]["top"], top)
        lines[ln]["right"]  = max(lines[ln]["right"], left+w)
        lines[ln]["bottom"] = max(lines[ln]["bottom"], top+h)
    # sort by top, then left
    ordered = sorted(lines.values(), key=lambda L: (L["top"], L["left"]))
    for L in ordered:
        L["text"] = " ".join(L["chunks"]).strip()
    return ordered

# --- 2) Classify lines ---
def classify_lines(lines):
    bullets, blocks = [], []
    eq_buf = []  # to merge consecutive equation lines

    def flush_eq():
        nonlocal eq_buf
        if eq_buf:
            # merge into aligned
            if len(eq_buf) == 1:
                blocks.append({"type":"equation", "latex": to_latex_safe(eq_buf[0]["text"])})
            else:
                body = " \\\\\n".join(to_latex_safe(L["text"]) for L in eq_buf)
                blocks.append({"type":"equation", "latex": "\\begin{aligned}\n" + body + "\n\\end{aligned}"})
            eq_buf = []

    for L in lines:
        txt = L["text"]

        # Heading?
        m = HEADING_RE.match(txt)
        if m:
            flush_eq()
            kind = m.group(1).lower()
            rest = to_latex_safe(m.group(2))
            blocks.append({"type": kind, "name":"", "content": rest})
            continue

        # Bullet?
        mb = BULLET_RE.match(txt)
        if mb:
            flush_eq()
            bullets.append(to_latex_safe(mb.group(2)))
            continue

        # Equation-like?
        if MATH_HINT.search(txt):
            eq_buf.append(L)
            continue

        # Plain text → bullet
        flush_eq()
        bullets.append(to_latex_safe(txt))

    flush_eq()
    return bullets, blocks

# --- 3) Render LaTeX ---
def render_latex(bullets, blocks, out_tex):
    tpl = Template(TEX_TEMPLATE)
    parts = ["\\section{Board Capture}"]
    if bullets:
        parts.append("\\begin{itemize}")
        for b in bullets:
            parts.append(f"\\item {b}")
        parts.append("\\end{itemize}")
    for blk in blocks:
        t = blk["type"]
        if t == "equation":
            parts.append("\\[\n" + blk["latex"] + "\n\\]")
        elif t in ("definition","theorem","lemma","proposition","corollary","remark","example"):
            name = blk.get("name","")
            maybe = f"[{name}]" if name else ""
            parts.append(f"\\begin{{{t}}}{maybe}\n{blk['content']}\n\\end{{{t}}}")
    tex = tpl.render(body="\n\n".join(parts))
    pathlib.Path(out_tex).write_text(tex)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="path to your (denoised) board image")
    ap.add_argument("--out", default="notes.tex")
    ap.add_argument("--pdf", action="store_true", help="run latexmk if available")
    args = ap.parse_args()

    lines = ocr_lines_with_boxes(args.image)
    bullets, blocks = classify_lines(lines)
    render_latex(bullets, blocks, args.out)

    if args.pdf:
        try:
            subprocess.check_call(["latexmk","-pdf","-silent",args.out])
        except FileNotFoundError:
            print("latexmk not found; wrote .tex only.", file=sys.stderr)

    print(f"✅ Wrote {args.out}")

if __name__ == "__main__":
    main()
