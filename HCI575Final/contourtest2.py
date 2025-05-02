import cv2
import numpy as np
import csv
import os

def build_transformation_matrix(center:tuple,
                                rotation_deg:float,
                                scale:float,
                                shear_deg:float):
    """
    Build a composite transformation matrix (3x3) that applies scaling, rotation, and shear about the center.

    Parameters:
      center (tuple): (x, y) center of the image.
      rotation_deg (float): Rotation angle in degrees.
      scale (float): Scaling factor.
      shear_deg (float): Shear angle in degrees (applied along x-axis).

    Returns:
      M (2x3 numpy array): Affine transformation matrix for cv2.warpAffine.
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

    # Shear matrix (shear along the x axis).
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


def transformation_generator(image: np.ndarray,
                             rotation_range,
                             scale_range,
                             shear_range):
    """
    Generator that yields affine transformation matrices and the resulting warped images.

    Parameters:
      image (np.ndarray): The input image.
      In this case the symbol used for templates and as a structural element.
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
    # Get the dimensions and center for input image
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

def match_template_with_transform(img_bin: np.ndarray,
                                  templates: list,
                                  threshold: float = 0.7) -> (bool, float):
    """
    Use normalized cross-correlation template matching to detect symbol.
    If the match is above threshold, return True. Otherwise, return best score.
    Best score used in

    Args:
        img_bin (np.ndarray): Binarized input image.
        templates (list): List of transformed templates.
        threshold (float): Score threshold for a match.

    Returns:
        match_found (bool): Whether a match was found.
        best_score (float): Best match score.
    """
    best_score = 0
    for tmpl in templates:
        res = cv2.matchTemplate(img_bin, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        best_score = max(best_score, max_val)

    match_found = best_score >= threshold
    return match_found, best_score

def match_symbol(img_bin: np.ndarray,
                 contours: list,
                 symbol_contour: np.ndarray,
                 templates: list,
                 log_to_csv: bool = False,
                 csv_path: str = "match_log.csv") -> list:
    """
    Composite symbol matching: Template match + contour-based score. Optionally logs details to CSV.

    Args:
        img_bin (np.ndarray): Binarized input image.
        contours (list): Detected contours from the image.
        symbol_contour (np.ndarray): Cleaned reference contour.
        templates (list): List of transformed templates.
        log_to_csv (bool): If True, writes each match score to CSV.
        csv_path (str): File path for CSV logging.

    Returns:
        matched_indices (list): Indices of matching contours.
    """

    matched_indices = []

    # Symbol vertices
    symbol_perimeter = cv2.arcLength(symbol_contour, True)
    symbol_approx = cv2.approxPolyDP(symbol_contour, 0.01 * symbol_perimeter, True)
    symbol_vertex_count = len(symbol_approx)

    # Symbol aspect ratio
    symbol_x, symbol_y, symbol_w, symbol_h = cv2.boundingRect(symbol_contour)
    symbol_aspect_ratio = symbol_w / symbol_h if symbol_h != 0 else 0

    #
    match_found, template_score = match_template_with_transform(img_bin, templates, threshold=0.7)

    rows = []

    for idx, cnt in enumerate(contours):
        match_score = cv2.matchShapes(symbol_contour, cnt, cv2.CONTOURS_MATCH_I1, 0.0)
        cnt_perimeter = cv2.arcLength(cnt, True)
        cnt_approx = cv2.approxPolyDP(cnt, 0.01 * cnt_perimeter, True)
        vertex_diff = abs(len(cnt_approx) - symbol_vertex_count)
        vertex_score = vertex_diff / symbol_vertex_count

        x, y, w, h = cv2.boundingRect(cnt)
        cnt_aspect_ratio = w / h if h != 0 else 0
        aspect_diff = abs(cnt_aspect_ratio - symbol_aspect_ratio)

        # If the template score is strong, trust it completely
        if match_found:
            composite_score = 0  # Considered a perfect match
        else:
            composite_score = (0.2 * match_score) + (0.65 * vertex_score) + (0.15 * aspect_diff) + (0.1 * (1 - template_score))

        print((f"Contour #{idx} - match: {match_score:.3f}, vertex_diff: {vertex_diff}, "
               f"aspect_diff: {aspect_diff:.3f}, template: {template_score:.3f}, composite: {composite_score:.3f}"))

        if composite_score < 0.3:
            print(f"Symbol matched at contour #{idx}!")
            matched_indices.append(idx)

        if log_to_csv:
            rows.append([idx, match_score, vertex_diff, vertex_score, aspect_diff, template_score, composite_score])

    if log_to_csv:
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["index", "match_score", "vertex_diff", "vertex_score", "aspect_diff", "template_score", "composite_score"])
            writer.writerows(rows)

    return matched_indices

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

def needs_smoothing(image: np.ndarray, threshold: float=0.01):
    """
    Determines if the input image needs smoothing by detecting noise and artifacts.
    Ended up being necessary since some of the memes are artifacted jpegs.

    Args:
        image (np.ndarray):
        threshold:
    Returns:

    """
    # Detect if the image has a lot of noise/artifacts
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    return laplacian_var > threshold * 1000

def smooth_image(image):
    # Basic smoothing steps
    blurred = cv2.GaussianBlur(image, (5, 5), 0)  # Smooth noise
    denoised = cv2.fastNlMeansDenoisingColored(blurred, None, 10, 10, 7, 21) if len(image.shape) == 3 else cv2.fastNlMeansDenoising(blurred, None, 10, 7, 21)
    return denoised


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
    if needs_smoothing(img):
        img = smooth_image(img)


    # Compute mean intensity to decide thresholding method due to variance in meme brightness
    mean_intensity = np.mean(img)

    if mean_intensity < 136:
        # Low-light image: Use adaptive thresholding
        img_bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 3)
    elif mean_intensity > 170:
        # Very bright image: Use binary inverse thresholding
        _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    else:
        # Mid-range contrast: Use Otsu's thresholding
        _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    # # Morphological opening to remove small white speckles
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel, iterations=1)
    return img_bin

test = load_structural_elements("se_dir")

swastika_list = test['swastika']
testMeme = preprocess_meme('images/0000003.jpg')
test1, test2 = match_template(testMeme, swastika_list)
print(f"Match Template{test1, test2}")