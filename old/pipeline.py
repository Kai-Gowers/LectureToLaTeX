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
    "You will be given rough OCR output from a blackboard in an abstract algebra lecture. "
    "The lecture concerns *groups, subgroups, normal subgroups, quotient groups, cosets, "
    "normalizers, conjugation,* and similar topics in group theory.\n\n"

    "Your job is to:\n"
    "1) Clean up the OCR and reconstruct the mathematics faithfully, and\n"
    "2) Add short, accurate explanations ONLY in the context of group theory "
    "(groups, subgroups, normality, quotient maps, cosets, normalizers, homomorphisms, etc.).\n\n"

    "=== CRITICAL CONSTRAINTS ===\n"
    "• You must ONLY discuss mathematical content that actually appears in the OCR.\n"
    "• Do NOT introduce unrelated topics such as fields, Galois theory, cyclotomic fields, "
    "L-functions, number theory, etc., unless those exact words appear in the OCR text.\n"
    "• Do NOT invent new topics or reinterpret the lecture. Explain ONLY what the board shows.\n"
    "• Your explanations must be tied strictly to the visible symbols, equations, and steps.\n\n"

    "=== LATEX RULES ===\n"
    "• Output a complete LaTeX document from \\documentclass to \\end{document}.\n"
    "• Use: \\documentclass[12pt]{article} and "
    "\\usepackage{amsmath,amssymb,amsfonts,amsthm}.\n"
    "• Every mathematical expression MUST be in math mode:\n"
    "    – Inline math: \\( ... \\)\n"
    "    – Display math: \\[ ... \\]\n"
    "• Use correct LaTeX operators: \\ker, \\operatorname{Im}, \\operatorname{Norm}, "
    "\\trianglelefteq, \\mathbb{Z}, \\mathbb{Q}, \\leq, etc.\n"
    "• Do NOT output markdown code fences. ONLY pure LaTeX.\n"
    "• No raw ASCII math (e.g., x^2, gKg^{-1}, a/b). Convert all of it to proper LaTeX.\n\n"

    "=== DOCUMENT STRUCTURE ===\n"
    "• Organize content using sections/subsections only if appropriate for the OCR.\n"
    "• Provide short, clear explanations immediately before or after important formulas.\n"
    "• Style should resemble a clean algebra/group theory lecture—not a textbook chapter.\n\n"

    "=== IF OCR IS UNCLEAR ===\n"
    "• Make the closest reasonable guess and add a LaTeX comment '% unclear'.\n\n"

    "Output ONLY the LaTeX document. No commentary or markdown."
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
    "without introducing unrelated topics like field extensions or cyclotomic fields.\n\n"
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
