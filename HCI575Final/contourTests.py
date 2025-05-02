import cv2
import numpy as np
from matplotlib.image import imread


def display_image(image: np.ndarray):
    """
    This exists mostly for debugging
    """
    cv2.imshow('Display', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def needs_smoothing(image, threshold=0.01):
    # Detect if the image has a lot of noise/artifacts
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    return laplacian_var > threshold * 1000

def smooth_image(image):
    # Basic smoothing steps
    blurred = cv2.GaussianBlur(image, (5, 5), 0)  # Smooth noise
    denoised = cv2.fastNlMeansDenoisingColored(blurred, None, 10, 10, 7, 21) if len(image.shape) == 3 else cv2.fastNlMeansDenoising(blurred, None, 10, 7, 21)
    return denoised

def preprocess_image(image):
    if needs_smoothing(image):
        print("Smoothing applied.")
        return smooth_image(image)
    else:
        print("No smoothing needed.")
        return image


swastika_se_path = "se_dir/swastika_se.png"
swastika_se = cv2.imread(swastika_se_path,cv2.IMREAD_GRAYSCALE)
_, thresh_se = cv2.threshold(swastika_se, 127, 255, cv2.THRESH_BINARY)


# Symbol contour
contours_se, hierarchy_se = cv2.findContours(thresh_se, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

swastika_se_copy = swastika_se.copy()
contoured_se = cv2.drawContours(swastika_se_copy, contours_se, -1, (100,100,100),2)
#display_image(contoured_se)


img_path = "images/0000001.png"
img_og = cv2.imread(img_path)

img = preprocess_image(img_og)
#display_image(img)

img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# May need extensive tweaking
# Apply Gaussian blur to reduce noise while preserving edges as medianBlur was ineffective
img_blur = cv2.GaussianBlur(img_grey, (5, 5), 0)

# Compute mean intensity to decide thresholding method due to variance in meme brightness
mean_intensity = np.mean(img_blur)

if mean_intensity < 136:
    # Low-light image: Use adaptive thresholding
    img_bin = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 3)
elif mean_intensity > 170:
    # Very bright image: Use binary inverse thresholding
    _, img_bin = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY_INV)
else:
    # Mid-range contrast: Use Otsu's thresholding
    _, img_bin = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#display_image(img_bin)
contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contoured_img = cv2.drawContours(img, contours, 47, (0,255,0),3)

display_image(img)

matched_indices = []  # To store indices of matching contours

for idx, cnt in enumerate(contours):
    match_score = cv2.matchShapes(contours_se[0], cnt, cv2.CONTOURS_MATCH_I1, 0.0)
    print(f"Contour #{idx} match score: {match_score}")
    if match_score < 0.1:  # Threshold for a "good enough" match
        print(f"Symbol found at contour #{idx}!")
        matched_indices.append(idx)  # Save index

print(f"Matched contour indices: {matched_indices}")

symbol_contour = ...  # your cleaned template contour

matched_indices = []

symbol_perimeter = cv2.arcLength(contours_se[0], True)
symbol_approx = cv2.approxPolyDP(contours_se[0], 0.01 * symbol_perimeter, True)
symbol_vertex_count = len(symbol_approx)

for idx, cnt in enumerate(contours):
    # Match scores
    match_score = cv2.matchShapes(contours_se[0], cnt, cv2.CONTOURS_MATCH_I1, 0.0)

    # Vertices
    cnt_perimeter = cv2.arcLength(cnt, True)
    cnt_approx = cv2.approxPolyDP(cnt, 0.01 * cnt_perimeter, True)
    vertex_diff = abs(len(cnt_approx) - symbol_vertex_count)

    # Normalize vertex_diff
    vertex_score = vertex_diff / symbol_vertex_count

    # Aspect ratio
    x, y, w, h = cv2.boundingRect(cnt)
    cnt_aspect_ratio = w / h if h != 0 else 0

    symbol_x, symbol_y, symbol_w, symbol_h = cv2.boundingRect(contours_se[0])
    symbol_aspect_ratio = symbol_w / symbol_h if symbol_h != 0 else 0

    aspect_diff = abs(cnt_aspect_ratio - symbol_aspect_ratio)

    # Composite score: weighted sum (tune weights as needed)
    composite_score = (0.2 * match_score) + (0.8 * vertex_score) + (0.15 * aspect_diff)

    print((f"Contour #{idx} - match: {match_score:.3f}, vertex_diff: {vertex_diff},"
           f"aspect_diff: {aspect_diff:.3f}, composite: {composite_score:.3f}"))

    if composite_score < 0.3:  # Looser threshold for noisy/hand-drawn input
        print(f"Symbol matched at contour #{idx}!")
        matched_indices.append(idx)

print(f"Matched indices: {matched_indices}")



exit()

import csv

# Prepare CSV file
csv_filename = "contour_data.csv"
fieldnames = [
    "Index", "Area", "Perimeter", "Approx_Vertices",
    "BoundingBox_X", "BoundingBox_Y", "BoundingBox_W", "BoundingBox_H",
    "Aspect_Ratio", "Centroid_X", "Centroid_Y", "Is_Convex", "ConvexHull_Points"
]

with open(csv_filename, mode='w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for idx, cnt in enumerate(contours_se):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.01 * perimeter, True)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h if h != 0 else 0

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        is_convex = cv2.isContourConvex(cnt)
        convex_hull = cv2.convexHull(cnt)

        # Save to CSV
        writer.writerow({
            "Index": idx,
            "Area": area,
            "Perimeter": perimeter,
            "Approx_Vertices": len(approx),
            "BoundingBox_X": x,
            "BoundingBox_Y": y,
            "BoundingBox_W": w,
            "BoundingBox_H": h,
            "Aspect_Ratio": round(aspect_ratio, 2),
            "Centroid_X": cx,
            "Centroid_Y": cy,
            "Is_Convex": int(is_convex),
            "ConvexHull_Points": len(convex_hull)
        })

        # Optional: Draw and show
        temp_img = swastika_se.copy()
        cv2.drawContours(temp_img, contours, idx, (255, 0, 0), 2)
        cv2.imshow("Contour Info", temp_img)
        key = cv2.waitKey(0)  # Wait for keypress before next
        if key == 27:  # ESC key to break early
            break

cv2.destroyAllWindows()

print(f"Contour data saved to {csv_filename}")



