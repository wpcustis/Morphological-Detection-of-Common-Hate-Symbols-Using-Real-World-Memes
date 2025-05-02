
import os
import cv2
import csv
import numpy as np
from contour3 import preprocess_meme, match_symbol



def detect_folder(images_dir: str,
                  se_dict: dict,
                  output_csv: str,
                  threshold: float):

    # Write header
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'present',"template_score", "composite_score"])

        for fname in sorted(os.listdir(images_dir)):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                continue

            img_path = os.path.join(images_dir, fname)
            img = cv2.imread(img_path)

            # Binarize / preprocess
            img_bin = preprocess_meme(img)

            # Check each symbol; stop at first found
            present = 0
            for name, (se_mask, se_contour) in se_dict.items():
                matches, template_score, composite_score = match_symbol(
                    img_bin,
                    se_mask,
                    se_contour,
                    # You can customize rotation/shear ranges here:
                    rotation_range=range(0, 90, 15),
                    shear_range=range(0, 6, 1),
                    log_to_csv=False
                )
                if matches:
                    present = 1
                    break

            writer.writerow([fname, present, template_score, composite_score])
            print(f"{fname}: {'FOUND' if present else 'not found'}")

    print(f"Detection complete. Results written to {output_csv}")

if __name__ == '__main__':
    images_dir = "images"

    swastika_mask = cv2.imread("se_dir/swastika_se.png", cv2.IMREAD_GRAYSCALE)
    _, swastika_mask = cv2.threshold(swastika_mask, 127, 255, cv2.THRESH_BINARY_INV)
    ss_mask = cv2.imread("se_dir/ss_se.png", cv2.IMREAD_GRAYSCALE)
    _, ss_mask = cv2.threshold(ss_mask, 127, 255, cv2.THRESH_BINARY_INV)

    swastika_contour = np.load('contours/swastika_contour.npy')
    ss_contour = np.load('contours/ss_contour.npy')
    se_dict = {"swastika": (swastika_mask,swastika_contour),
              "ss": (ss_mask, ss_contour)}
    output_csv = "output"
    threshold = 0.8
    detect_folder(images_dir, se_dict, output_csv, threshold)
