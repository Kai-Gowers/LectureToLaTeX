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