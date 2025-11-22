import os
import subprocess
from openai import OpenAI
from denoise_pipeline import run_denoise
import pytesseract
from PIL import Image

# =============== CONFIG ===============
DOCS_DIR = "notes_out"       
MODEL_NAME = "deepseek-chat"
API_KEY = os.environ.get("DEEPSEEK_API_KEY") or "sk-your-key-here"
BASE_URL = "https://api.deepseek.com"
# ======================================

os.makedirs(DOCS_DIR, exist_ok=True)

print("[INFO] Running denoise pipeline on image from raw/ ...")
paths = run_denoise()   # PASS IN IMAGE OF YOUR CHOICE AS PARAMETER in_path="raw/some_other.jpg" 
enh_path = paths["enhanced"]

image_base = paths['base_name']
note_name = f"notes_{image_base}"
print(f"[INFO] Using enhanced image for OCR: {enh_path}")

# ===== 1) OCR the processed image =====
ocr_text = pytesseract.image_to_string(Image.open(enh_path))
print(ocr_text)
print("[INFO] OCR text extracted.")

# ===== 2) Call LLM with text only =====
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

system_prompt = (
    "You are a LaTeX math transcription AND explanation assistant. "
    "You will be given rough OCR output from a handwritten mathematics blackboard. "
    "Your task is to clean it, reconstruct the mathematics faithfully, and "
    "produce a polished, structured LaTeX article with detailed explanations.\n\n"

    "=== CORE TASKS ===\n"
    "1. Correct the OCR output and rewrite all mathematical content in proper LaTeX.\n"
    "2. Add clear explanatory text (in full sentences) before or after each major step, "
    "suitable for an advanced undergraduate or beginning graduate student.\n"
    "3. Preserve all important equations, derivations, and logical structure.\n\n"

    "=== STRICT LATEX RULES ===\n"
    "• Every mathematical symbol or expression MUST be in math mode.\n"
    "  – Inline math → \\( ... \\)\n"
    "  – Display math → \\[ ... \\]\n"
    "• Never leave raw math symbols in text (e.g. x^2, sum, int, a/b).\n"
    "• Use correct LaTeX operators: \\ker, \\operatorname{Gal}, \\Hom, \\bQ, \\bZ, \\mod, \\leq, etc.\n"
    "• Use standard formatting for groups, fields, cosets, cyclotomic extensions, etc.\n"
    "• No markdown code fences. ONLY pure LaTeX.\n\n"

    "=== DOCUMENT STRUCTURE ===\n"
    "• Output a complete LaTeX document:\n"
    "  \\documentclass[12pt]{article}\n"
    "  \\usepackage{amsmath, amssymb, amsfonts, amsthm}\n"
    "  ...\n"
    "  \\begin{document}\n"
    "  ... content ...\n"
    "  \\end{document}\n"
    "• Use sections, subsections, and paragraphs to organize the material.\n"
    "• You may use environments such as theorem, definition, remark, proof, itemize, or enumerate.\n"
    "• Explanations must also follow the strict math-mode rules when referencing symbols.\n\n"

    "=== STYLE REQUIREMENTS ===\n"
    "Your output should resemble a clean textbook or research monograph style similar to "
    "graduate-level algebraic number theory literature. "
    "Ensure consistent math-mode usage, operator spacing, and paragraph structure.\n\n"

    "If something from the OCR is ambiguous or unreadable, include a LaTeX comment '% unclear'.\n"
    "Output ONLY LaTeX, with no markdown and no external commentary."
)



user_prompt = (
    "Here is the raw OCR output from a math blackboard image. "
    "The OCR may have mistakes, missing backslashes, or broken fractions. "
    "Please:\n"
    "• Rewrite it as clean, correct LaTeX (article class + amsmath), and\n"
    "• Insert detailed explanations and commentary in LaTeX so that a reader can follow the reasoning.\n"
    "\n"
    "You should keep the original mathematical content and derivations, but you are encouraged to:\n"
    "• Organize the material with sections/subsections,\n"
    "• Add short explanatory paragraphs around each important formula or step, and\n"
    "• Clarify the meaning of symbols and assumptions when they are implicit.\n"
    "\n"
    "Remember to output only LaTeX (no markdown) and to make it fully compilable.\n\n"
    f"OCR START:\n{ocr_text}\nOCR END."
)


response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    stream=False,
)

latex_source = response.choices[0].message.content
print("[INFO] LLM returned LaTeX.")

if latex_source.strip().startswith("```"):
    latex_source = latex_source.strip().strip("`")

tex_path = os.path.join(DOCS_DIR, f"{note_name}.tex")
with open(tex_path, "w") as f:
    f.write(latex_source)

print(f"[INFO] Wrote LaTeX to {tex_path}")

# ===== 3) Compile to PDF =====
try:
    subprocess.run(
        ["latexmk", "-pdf", f"{note_name}.tex", f"-outdir={DOCS_DIR}"],
        check=True,
        cwd=DOCS_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print(f"[INFO] PDF generated → {os.path.join(DOCS_DIR, f'{note_name}.pdf')}")
except FileNotFoundError:
    print("[WARN] latexmk not found, trying pdflatex...")
    subprocess.run(
        ["pdflatex", f"{note_name}.tex"],
        check=True,
        cwd=DOCS_DIR,
    )
    print(f"[INFO] PDF generated → {os.path.join(DOCS_DIR, f'{note_name}.pdf')}")
except subprocess.CalledProcessError as e:
    print("[ERROR] LaTeX compilation failed.")
    print(e.stdout.decode("utf-8", errors="ignore"))
    print(e.stderr.decode("utf-8", errors="ignore"))
