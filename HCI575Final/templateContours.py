import cv2
import csv
import numpy as np
from contour3 import preprocess_meme
from contour3 import find_filter_contours




# Helper file to generate a csv for contour data from memes and templates for tuning.


# Prepare CSV file
csv_filename = "filtered_contour_data_0092.csv"
fieldnames = [
    "Index", "Area", "Perimeter", "Approx_Vertices",
    "BoundingBox_X", "BoundingBox_Y", "BoundingBox_W", "BoundingBox_H",
    "Aspect_Ratio", "Centroid_X", "Centroid_Y", "Is_Convex", "ConvexHull_Points"
]

# swastika_se_path = "se_dir/swastika_se.png"
# swastika_se = cv2.imread(swastika_se_path,cv2.IMREAD_GRAYSCALE)
# _, thresh_swastika_se = cv2.threshold(swastika_se, 127, 255, cv2.THRESH_BINARY)

# # Symbol contour
# contours_swastika_se, hierarchy_se = cv2.findContours(thresh_swastika_se, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# swastika_contour = contours_swastika_se[0]
#
# # Saved contours for later use
# np.save('swastika_contour.npy', swastika_contour)
#
# ss_se_path = "se_dir/ss_se.png"
# ss_se = cv2.imread(ss_se_path,cv2.IMREAD_GRAYSCALE)
# _, thresh_ss_se = cv2.threshold(ss_se, 127, 255, cv2.THRESH_BINARY)
#
# # Symbol contour
# contours_ss_se, hierarchy_se = cv2.findContours(thresh_ss_se, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# ss_contour = contours_ss_se[0]

# # Saved contours for later use
# np.save('ss_contour.npy', ss_contour)

image = cv2.imread('images/hateful_0092.png')

contours_01 = find_filter_contours(image,(0.66,2),(15,25),False, True)

with open(csv_filename, mode='w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for idx, cnt in enumerate(contours_01):
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

        # Draw and show
        temp_img = image.copy()
        cv2.drawContours(temp_img, contours_01, idx, (100, 255, 0), 2)
        cv2.imshow(f"Contour Info {idx}", temp_img)
        key = cv2.waitKey(0)
        if key == 27:  # ESC key to break early
            break

cv2.destroyAllWindows()