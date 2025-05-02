import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt
import argparse

def display_images_from_dict(image_dict):
    """
    This exists mostly for debugging
    """
    for key, values in image_dict.items():
        # Have to grab image out of dict values
        for img in values:
            type(img)
            if type(img) is not np.ndarray:
                print(f'None np.ndarray skipped:{type(img)}')
                continue
            cv2.imshow(key, img)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

def display_image(image: np.ndarray):
    """
    This exists mostly for debugging, most functions just read out the 3 lines you need
    """
    cv2.imshow('Display', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def build_transformation_matrix(center, rotation_deg, scale, shear_deg):
    """
    Build a composite transformation matrix that applies scaling, rotation, and shear about the center.

    Args:
      center (tuple): (x, y) center of the image.
      rotation_deg (float): Rotation angle in degrees.
      scale (float): Scaling factor.
      shear_deg (float): Shear angle in degrees (applied along x-axis).

    Returns:
      M_full (2x3 np.ndarray): Affine transformation matrix for cv2.warpAffine.
    """
    # Convert angles from degrees to radians.
    angle = np.deg2rad(rotation_deg)
    shear = np.deg2rad(shear_deg)

    # Translation matrices to move to/from center.
    T1 = np.array([[1, 0, -center[0]],
                   [0, 1, -center[1]],
                   [0, 0, 1]])
    T2 = np.array([[1, 0, center[0]],
                   [0, 1, center[1]],
                   [0, 0, 1]])

    # Scaling matrix.
    S = np.array([[scale, 0, 0],
                  [0, scale, 0],
                  [0, 0, 1]])

    # Rotation matrix.
    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle), np.cos(angle), 0],
                  [0, 0, 1]])

    # Shear matrix (shear along the x-axis).
    H = np.array([[1, np.tan(shear), 0],
                  [0, 1, 0],
                  [0, 0, 1]])

    # Compose the transformations:
    # 1. Shift to origin (T1)
    # 2. Scale, then rotate, then shear (S, R, H)
    # 3. Shift back to center (T2)
    M_full = T2 @ H @ R @ S @ T1  # using @ for matrix multiplication

    # Return the 2x3 matrix for cv2.warpAffine.
    return M_full[:2, :]


def transformation_generator(image, rotation_range, scale_range, shear_range):
    """
    Takes in images and affine transformation matrices and returns the resulting warped images.

    Args:
      image (numpy array): The input image.
      rotation_range (iterable): List or range of rotation angles in degrees.
      scale_range (iterable): List or range of scaling factors.
      shear_range (iterable): List or range of shear angles in degrees.

    Yields:
      tuple: (M, warped, rotation, scale, shear)
        M         - The 2x3 affine transformation matrix.
        warped    - The resulting warped image.
        rotation  - The rotation angle used.
        scale     - The scaling factor used.
        shear     - The shear angle used.
    """
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    for rotation in rotation_range:
        for scale in scale_range:
            for shear in shear_range:
                # Build the transformation matrix.
                M = build_transformation_matrix(center, rotation, scale, shear)

                # Compute new output image dimensions by transforming the corners.
                corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
                # cv2.transform expects a shape (1, N, 2)
                transformed_corners = cv2.transform(np.array([corners]), M)[0]

                # Determine the bounding rectangle for the transformed corners.
                x, y, w_new, h_new = cv2.boundingRect(transformed_corners.astype(np.int32))

                # Adjust the transformation matrix to account for translation so the entire image is visible.
                M_adjusted = M.copy()
                M_adjusted[:, 2] -= [x, y]

                # Apply the warp.
                warped = cv2.warpAffine(image, M_adjusted, (w_new, h_new))
                yield M_adjusted, warped, rotation, scale, shear

def load_structural_elements(se_dir: str) -> dict:
    """
    Load SE images (binary masks) and generate rotated variants.
    Assumes filenames 'swastika_se.png' and 'ss_se.png' in `se_dir`.

    Args:
        se_dir (string): Path to SE image containing folder.
    Returns:
        se_templates (dict): dict containing the se variants.

    images collected from the ADL's Hate on Display resource and modified via Photoshop
    SS Bolts: https://www.adl.org/resources/hate-symbol/ss-bolts
    Swastika: https://www.adl.org/resources/hate-symbol/swastika
    """
    # Grab original SE images
    files = {
        'swastika': os.path.join(se_dir, 'swastika_se.png'),
        'ss': os.path.join(se_dir, 'ss_se.png')
    }

    # Create dict of SE variants.
    se_templates = {}
    for key, path in files.items():
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Structural element not found: {path}")
        # Binarize template
        _, se_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        # Generate
        # need to look at transformation generator and change this next line to a for loop
        # if I need other info
        se_templates[key] = [warped for _, warped, _, _, _ in transformation_generator(
            se_bin, rotation_range=range(0,90,15), scale_range=[1.0], shear_range=[0])]
    return se_templates

test = load_structural_elements("se_dir")
print(len(test['ss']))
# for i in range(0,len(test['ss'])):
#     print(i)
#     print(test['ss'][i])

def preprocess_meme(meme_path: str) -> np.ndarray:
    """
    Load and binarize an input meme for symbol detection.
    Uses adaptive thresholding for robustness.

    """
    # load in meme
    img = cv2.imread(meme_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {meme_path}")


    # May need extensive tweaking
    # Apply Gaussian blur to reduce noise while preserving edges as medianBlur was ineffective
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)

    # Compute mean intensity to decide thresholding method due to variance in meme brightness
    mean_intensity = np.mean(img_blur)
    print(f"Mean Intensity: {mean_intensity}")
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


    # # Morphological opening to remove small white speckles
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel, iterations=1)
    return img_bin

# In testing, there is considerable issues with non-standard images even with the correct orientation
# Consider 0000001.png results

testMeme = preprocess_meme('images/hateful_0067.png')
display_image(testMeme)

# image 0000001.png tests
# mean brightness for color 124
# mean brightness for grayscale 134

# testMeme = cv2.imread('images/0000001.png', cv2.IMREAD_GRAYSCALE)
# testimg_blur = cv2.GaussianBlur(testMeme, (5, 5), 0)
#
# # Compute mean intensity to decide thresholding method due to variance in meme brightness
# testmean_intensity = np.mean(testimg_blur)
# print(testmean_intensity)

# Below is taken from OpenCV's documentation
# All the 6 methods for comparison in a list
methods = ['TM_CCOEFF', 'TM_CCOEFF_NORMED', 'TM_CCORR',
            'TM_CCORR_NORMED', 'TM_SQDIFF', 'TM_SQDIFF_NORMED']


img = testMeme
assert img is not None, "file could not be read, check with os.path.exists()"
img2 = img.copy()
template = test['swastika'][3] # 45 degree
# _, template = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY_INV)
display_image(template)
assert template is not None, "file could not be read, check with os.path.exists()"
w, h = template.shape[::-1]

for meth in methods:
    img = img2.copy()
    method = getattr(cv2, meth)

    # Apply template Matching
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img, top_left, bottom_right, 255, 2)

    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)

    plt.show()


def match_template(img_bin: np.ndarray, templates: list, threshold: float = 0.7) -> (bool, float):
    """
    Use normalized cross-correlation template matching to detect symbol.
    Need to add some of the generator parameters here so I know what got the best score?

    Args:
        img_bin (np.ndarray): Binarized input image (meme)

    Returns:

        best_score (float): score of the best scoring item in templates
    """
    best_score = 0
    for tmpl in templates:
        res = cv2.matchTemplate(img_bin, tmpl, cv2.TM_SQDIFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        best_score = max(best_score, max_val)
        match_found = best_score >= threshold
    return match_found, best_score

swastika_list = test['swastika']
test1, test2 = match_template(testMeme, swastika_list)
print(f"Match Template{test1, test2}")
