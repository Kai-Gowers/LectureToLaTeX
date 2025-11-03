import cv2
import os
import numpy as np

def enhance_chalkboard(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(th) < 127:
        th = cv2.bitwise_not(th)
    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(blur, 50, 150)
    return closed, edges


def run_denoise(
    in_path="raw/TestImage2.jpeg",
    processed_dir="processed"
):
    os.makedirs(processed_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(in_path))[0] 

    # ---- read the raw image ----
    img = cv2.imread(in_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Couldn't read {in_path} - did you put the image in raw/?")

    # ---- denoise ----
    denoise_strength = 10
    img_dn = cv2.fastNlMeansDenoisingColored(img, None, denoise_strength, denoise_strength, 7, 21)
    denoised_path = os.path.join(processed_dir, f"{base_name}_denoised.jpg")
    cv2.imwrite(denoised_path, img_dn)
    print(f"Saved denoised image → {denoised_path}")

    # ---- enhance + edges ----
    enh, edg = enhance_chalkboard(img_dn)
    enh_path = os.path.join(processed_dir, f"{base_name}_enhanced.jpg")
    edg_path = os.path.join(processed_dir, f"{base_name}_edges.jpg")
    cv2.imwrite(enh_path, enh)
    cv2.imwrite(edg_path, edg)
    print(f"Saved enhanced image → {enh_path}")
    print(f"Saved edges image → {edg_path}")

    # ---- return paths so pipeline.py can use the correct names ----
    return {
        "base_name": base_name,
        "denoised": denoised_path,
        "enhanced": enh_path,
        "edges": edg_path,
    }

if __name__ == "__main__":
    run_denoise()
