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

IN_DIR = "pre_out"
OUT_DIR = "pre_out"
MAX_SIDE = 2000

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
        h, w = img.shape[:2]
        long_side = max(h, w)
        if long_side > MAX_SIDE:
            scale = MAX_SIDE / long_side
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(OUT_DIR, f"rescaled_{filename}"), img)
        print(f"Processed {filename}")
print("All images saved in:", OUT_DIR)
