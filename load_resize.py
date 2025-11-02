import cv2
import os

IN_PATH = "/Users/kai/Downloads/SampleNotes.jpg" #CHANGE TO IMAGE FILE NAME !!!
OUT_DIR = "pre_out"
MAX_SIDE = 2000

os.makedirs(OUT_DIR, exist_ok=True)

img = cv2.imread(IN_PATH, cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError(f"Couldn't read {IN_PATH}")
cv2.imwrite(os.path.join(OUT_DIR, "00_original.jpg"), img)

h, w = img.shape[:2]
long_side = max(h, w)
if long_side > MAX_SIDE:
    scale = MAX_SIDE / long_side
    new_w, new_h = int(w * scale), int(h * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
cv2.imwrite(os.path.join(OUT_DIR, "01_resized.jpg"), img)

print("Saved:")
print("pre_out/00_original.jpg")
print("pre_out/01_resized.jpg")

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

IN_ZIP = "/Users/alexandra/Downloads/SampleNotesAS.zip"
OUT_DIR = "pre_out"
MAX_SIDE = 2000

with zipfile.ZipFile(IN_ZIP, "r") as zip_ref:
    extract_dir = IN_ZIP.replace(".zip", "")
    zip_ref.extractall(extract_dir)

os.makedirs(OUT_DIR, exist_ok=True)

for filename in os.listdir(extract_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        in_path = os.path.join(extract_dir, filename)
        img = cv2.imread(in_path, cv2.IMREAD_COLOR)
        if img is None:
            continue

        cv2.imwrite(os.path.join(OUT_DIR, f"00_original_{filename}"), img)

        h, w = img.shape[:2]
        long_side = max(h, w)
        if long_side > MAX_SIDE:
            scale = MAX_SIDE / long_side
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        cv2.imwrite(os.path.join(OUT_DIR, f"01_resized_{filename}"), img)

        enh, edges = enhance_chalkboard(img)
        cv2.imwrite(os.path.join(OUT_DIR, f"02_enhanced_{filename}"), enh)
        cv2.imwrite(os.path.join(OUT_DIR, f"03_edges_{filename}"), edges)

        print(f"Processed {filename}")

print("All images processed and saved in:", OUT_DIR)
