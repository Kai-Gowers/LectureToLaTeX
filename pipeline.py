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
    "You are a LaTeX math transcription assistant. "
    "You will be given a rough OCR extraction of a handwritten math blackboard. "
    "Your job is to clean it up and produce a full, compilable LaTeX document. "
    "Infer math symbols where obvious (e.g. 'sum' -> \\sum, 'int' -> \\int, '^' -> superscript). "
    "If something is unclear, add a LaTeX comment like '% unclear'. "
    "Output ONLY LaTeX, no markdown code fences."
)

user_prompt = (
    "Here is the raw OCR output from a math blackboard image. "
    "The OCR may have mistakes, missing backslashes, or broken fractions. "
    "Please rewrite it as clean LaTeX, using article class and amsmath.\n\n"
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
