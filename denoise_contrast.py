import cv2
import os

IN_PATH = "pre_out/01_resized.jpg"  
OUT_DIR = "pre_out"
os.makedirs(OUT_DIR, exist_ok=True)

img = cv2.imread(IN_PATH, cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError(f"Couldn't read {IN_PATH} - did you run Step 1?")

denoise_strength = 10
img_dn = cv2.fastNlMeansDenoisingColored(img, None, denoise_strength, denoise_strength, 7, 21)
cv2.imwrite(os.path.join(OUT_DIR, "02_denoise.jpg"), img_dn)

print("Saved:")
print("pre_out/02_denoise.jpg  (gentle denoise)")



import numpy as np

def enhance_chalkboard(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(th) < 127:
        th = cv2.bitwise_not(th)
    kernel = np.ones((2,2), np.uint8)
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(blur, 50, 150)
    return closed, edges

IN_DIR = "pre_out"
OUT_DIR = "pre_out_edges"

os.makedirs(OUT_DIR, exist_ok=True)
for root, _, files in os.walk(IN_DIR):
    for filename in files:
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        if filename.startswith(("00_", "01_", "02_", "03_")):
            continue
        in_path = os.path.join(root, filename)
        img = cv2.imread(in_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        enh, edg = enhance_chalkboard(img)
        cv2.imwrite(os.path.join(OUT_DIR, f"enhanced_{filename}"), enh)
        cv2.imwrite(os.path.join(OUT_DIR, f"edges_{filename}"), edg)
        print(f"Processed {filename}")
print("All images saved in:", OUT_DIR)
