
import os
import cv2
import csv
import numpy as np
from detectionFunctions import match_symbol



def detect_folder(images_dir: str,
                  se_dict: dict,
                  output_csv: str):

    # Write header
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'match', "matches","template_score", "shape_score", "composite_score", "method"])

        for fname in sorted(os.listdir(images_dir)):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                continue

            img_path = os.path.join(images_dir, fname)
            img = cv2.imread(img_path)



            # Check each symbol; stop at first found
            for name, (se_mask, se_contour) in se_dict.items():
                csv_name = f"test_09/{fname}_match_log.csv"
                match, matches, template_score, shape_score, best_composite_score, method = match_symbol(img, se_mask,
                                                                                                 se_contour,
                                                                                                 rotation_range=range(0,
                                                                                                                      75,
                                                                                                                      15),
                                                                                                 shear_range=range(-2, 2,
                                                                                                                   2),
                                                                                                 log_to_csv=False,
                                                                                                 csv_path=csv_name)


            writer.writerow([fname, match, matches, template_score, shape_score, best_composite_score, method])
            print(f"{fname}: {'FOUND' if match else 'not found'}")

    print(f"Detection complete. Results written to {output_csv}")

if __name__ == '__main__':
    images_dir = "images"

    swastika_mask = cv2.imread("se_dir/swastika_se.png", cv2.IMREAD_GRAYSCALE)
    _, swastika_mask_inv = cv2.threshold(swastika_mask, 127, 255, cv2.THRESH_BINARY)
    _, swastika_mask = cv2.threshold(swastika_mask, 127, 255, cv2.THRESH_BINARY_INV)
    swastika_contour = np.load('contours/swastika_contour.npy')
    se_dict = {"swastika": (swastika_mask,swastika_contour),
               "swastika_inv": (swastika_mask_inv,swastika_contour)}

    current_test = "10_harmful"
    output_csv = f"output_test_{current_test}.csv"
    detect_folder(images_dir, se_dict, output_csv)
