# denoise_contrast.py

import cv2
import os
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

def run_denoise(in_path="pre_out/01_resized.jpg", pre_out="pre_out", out_dir="out_denoise"):
    os.makedirs(pre_out, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(in_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Couldn't read {in_path} - did you run Step 1?")

    # Denoise
    denoise_strength = 10
    img_dn = cv2.fastNlMeansDenoisingColored(img, None, denoise_strength, denoise_strength, 7, 21)
    denoised_path = os.path.join(pre_out, "02_denoise.jpg")
    cv2.imwrite(denoised_path, img_dn)
    print(f"Saved: {denoised_path} (gentle denoise)")

    # Enhance and detect edges
    enh, edg = enhance_chalkboard(img_dn)
    enh_path = os.path.join(out_dir, "enhanced_01_resized.jpg")
    edg_path = os.path.join(out_dir, "edges_01_resized.jpg")
    cv2.imwrite(enh_path, enh)
    cv2.imwrite(edg_path, edg)
    print(f"Processed image → {enh_path}")

    # Return useful paths
    return {
        "denoised": denoised_path,
        "enhanced": enh_path,
        "edges": edg_path
    }

# only auto-run if executed directly
if __name__ == "__main__":
    run_denoise()
