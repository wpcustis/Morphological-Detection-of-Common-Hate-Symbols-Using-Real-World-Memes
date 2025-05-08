import cv2
import numpy as np
import csv
import os


def display_image(image: np.ndarray):
    """
    This exists mostly for debugging
    """
    cv2.imshow('Display', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Meme related functions

def needs_smoothing(image: np.ndarray, threshold: float=0.01) -> bool:
    """
    Determines if the input image needs smoothing by detecting noise and artifacts.
    Ended up being necessary since some of the memes are artifacted jpegs.

    Args:
        image (np.ndarray):
        threshold (float):
    Returns:
        smooth_bool:
    """
    # Detect if the image has a lot of noise/artifacts
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    smooth_bool = laplacian_var > threshold * 1000
    return smooth_bool

def smooth_image(image):
    # Basic smoothing steps
    blurred = cv2.GaussianBlur(image, (3, 3), 0)  # Smooth noise
    denoised = cv2.fastNlMeansDenoisingColored(blurred, None, 10, 10, 7, 21) if len(image.shape) == 3 else cv2.fastNlMeansDenoising(blurred, None, 10, 7, 21)
    return denoised

def preprocess_meme(image: np.ndarray) -> np.ndarray:
    """
    Binarize an input meme for symbol detection.
    Uses intensity-based thresholding for robustness.

    NOTE! this is being changed over as I speak from path to image input

    Args:
        image (np.ndarray): Image/meme being preprocessed
    Returns:
        img_bin
    """

    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

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


    # Morphological closing then opening to remove small holes and speckles
    return img_bin

# Transformation functions

def build_transformation_matrix(center: tuple,
                                rotation_deg: float,
                                scale: float,
                                shear_deg: float):
    """
    Build a composite transformation matrix (3x3) that applies scaling, rotation, and shear about the center.

    Parameters:
      center (tuple): (x, y) center of the image.
      rotation_deg (float): Rotation angle in degrees.
      scale (float): Scaling factor, opted for isotropic scaling since distortion in image is unknown and
      anisotropic felt much more computationally intense.
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


def transformation_generator(image: np.ndarray,
                             rotation_range,
                             scale_range,
                             shear_range):
    """
    Generator that yields affine transformation matrices and the resulting warped images.

    Contains

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

    h_img, w_img = image.shape
    center = (w_img / 2, h_img / 2)

    for rotation in rotation_range:
        for scale in scale_range:
            for shear in shear_range:
                # Build the transformation matrix.
                M = build_transformation_matrix(center, rotation, scale, shear)

                # Compute new output image dimensions by transforming the corners.
                corners = np.array([[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]], dtype=np.float32)
                # cv2.transform expects a shape (1, N, 2)
                transformed_corners = cv2.transform(np.array([corners]), M)[0]

                # Determine the bounding rectangle for the transformed corners.
                x, y, w_new, h_new = cv2.boundingRect(transformed_corners.astype(np.int32))

                # Adjust the transformation matrix to account for translation so the entire image is visible.
                M_adjusted = M.copy()
                M_adjusted[:, 2] -= [x, y]

                # Apply the warp.
                warped = cv2.warpAffine(image, M_adjusted, (w_new, h_new), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                yield M_adjusted, warped, rotation, scale, shear

def find_filter_contours(image: np.ndarray,
                    aspect_range: tuple = (0.9, 1.6),
                    vertex_range: tuple = (17, 21),
                    convexity: bool = None,
                    remove_largest: bool = True,
                    approx_eps: float = 0.01) -> tuple:
    """
    Return only those contours that satisfy:
      - aspect_ratio = width / height within aspect_range
      - number of vertices within vertex_range
      - convexity matches
    In exploration for this project, these elements were the most distinct

    Args:
        image (np.ndarray): The binarized image
        aspect_range (tuple): Contains (min_aspect, max_aspect)
        vertex_range (tuple): (min_vertices, max_vertices)
        convexity (bool): True to require convex, False to require concave,
                        or None to accept both.
        approx_eps (float): fraction of perimeter to use for polygonal approx. Default = 0.01.
        remove_largest (bool): removes largest area object in contours since this uses cv2.RETR_LIST
        to grab all contours, and one inevitably will be the contour for the whole image.
        Defaults to True.

    Returns:
        filtered (tuple): A filtered tuple of contours that conform to the desired specs.
    """
    w, h = image.shape
    image_area = w * h
    area_thresh = 0.0005 * image_area

    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours_copy = list(contours)
    # find index of largest area contour

    # Remove largest area contour
    if remove_largest:
        # Remove the largest contour
        largest_idx = max(range(len(contours_copy)), key=lambda i: cv2.contourArea(contours_copy[i]))
        largest_contour = contours_copy[largest_idx]
        contours_copy.pop(largest_idx)


    filtered = []
    min_aspect, max_aspect = aspect_range
    min_verts, max_verts = vertex_range

    for cnt in contours_copy:
        # Compute bounding rect and aspect ratio
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            continue

        # remove small contours
        cnt_area = cv2.contourArea(cnt)

        if area_thresh > cnt_area:
            continue

        aspect = w / float(h)
        if not (min_aspect <= aspect <= max_aspect):
            continue

        # Polygonal approximation to count vertices
        peri = cv2.arcLength(cnt, True)
        epsilon = approx_eps * peri
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        n_verts = len(approx)
        if not (min_verts <= n_verts <= max_verts):
            continue

        # Convexity test
        is_convex = cv2.isContourConvex(cnt)
        if convexity is True and not is_convex:
            continue
        if convexity is False and is_convex:
            continue

        filtered.append(cnt)
    if not filtered:
        filtered.append(largest_contour)
    filtered = tuple(filtered)
    return filtered

def compute_scale_range(contours: tuple,
                        symbol_contour: np.ndarray,
                        margin: float = 0.2,
                        absolute_min: float = 0.05,
                        absolute_max: float = 5.0,
                        steps: int = 3) -> list[float]:
    """
    Compute an isotropic scale-range for template matching based on
    contour bounding‐box dimensions rather than pure area.

    Contained in match_symbol

    Args:
        contours: Tuple of contours detected in the scene.
        symbol_contour: The template contour.
        margin: Fractional expansion around the observed scale(s).
        absolute_min: Hard lower‐bound on the scale factor. Default = 0.05.
        absolute_max: Hard upper‐bound on the scale factor. Default = 5.
        steps: Number of discrete scale steps to generate.

    Returns:
        scale_list: A list of isotropic scale factors.
    """
    # Get template bounding‐box
    _, _, w_sym, h_sym = cv2.boundingRect(symbol_contour)

    # Compute per‐contour scales
    scales = []
    for cnt in contours:

        x, y, w, h = cv2.boundingRect(cnt)
        if w <= 0 or h <= 0:
            continue
        sx = w / w_sym
        sy = h / h_sym
        # Here I'm taking the average of the two axes for an isotropic guess;
        # Using the min of the ratios for width and height for isotropic scaling
        # Looked at min, max, and average, and
        s = min(sx,sy)
        scales.append(s)

    if not scales:
        # Fallback if no valid contours
        base_scales = [1.0]
    else:
        base_scales = scales

    # Determine observed min/max
    obs_min = min(base_scales)
    obs_max = max(base_scales)

    # Expand by margin just in case the additional margins in the base image are an issue
    lower = obs_min * (1 - margin)
    upper = obs_max * (1 + margin)

    # Clamp to absolutely sane bounds
    lower = max(lower, absolute_min)
    upper = min(upper, absolute_max)
    if lower >= upper:
        # If margin or clamping collapses range, nudge a bit around the median
        median = (obs_min + obs_max) / 2
        lower = max(median * (1 - margin), absolute_min)
        upper = min(median * (1 + margin), absolute_max)

    # Build and return the discrete list
    scale_list = list(np.linspace(lower, upper, num=steps))
    return scale_list



def detect_with_convolution(roi: np.ndarray, kernel: np.ndarray) -> int:
    """
    Convolve roi with kernel and return the total number of perfect‐match pixels.
    """
    # roi
    sum_k = int(kernel.sum())
    # filter2D does a full convolution
    conv = cv2.filter2D(roi, cv2.CV_32S, kernel)
    # count positions where the entire kernel fit exactly
    hits = np.count_nonzero(conv == sum_k)
    return hits


def morphological_detect(img_bin: np.ndarray,
                         kernels: list,
                         contours: list[np.ndarray],
                         pad: int = 5) -> bool:
    """
    Morphological hit-or-miss detection over a list of kernels.

    Returns True if *any* kernel yields between 1 and 4 hits (inclusive)
    Based on their not being more than 4 instances of the image per meme that I noticed.
    Otherwise returns False.

    Args:
        img_bin (np.ndarray): Binarized image (uint8, 255=fg).
        kernels (list): List of binarized templates (uint8, 255=fg).

    Returns:
        bool: True if a “goldilocks” match (1–4 hits) is found.
    """
    h_img, w_img = img_bin.shape

    for cnt in contours:
        # 1) get bounding box and pad it
        x, y, w, h = cv2.boundingRect(cnt)
        x0 = max(x - pad, 0)
        y0 = max(y - pad, 0)
        x1 = min(x + w + pad, w_img)
        y1 = min(y + h + pad, h_img)

        roi = img_bin[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        for kernel in kernels:
            kh, kw = kernel.shape
            # skip kernels too large to ever match
            if kh > roi.shape[0] or kw > roi.shape[1]:
                continue

            # perform hit-or-miss
            hitmiss = detect_with_convolution(roi, kernel)
            count = int(np.count_nonzero(hitmiss))

            # return True if this kernel has 1–4 hits
            if 1 <= count <= 4:
                return True

    return False



# Template matching functions

def match_template_with_transform(img_bin: np.ndarray,
                                  templates: list,
                                  threshold: float = 0.7) -> (bool, float):
    """
    Use normalized cross-correlation template matching to detect symbol.
    If the match is above threshold, return True. Otherwise, return best score.

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
        # ensure template not larger than image
        if tmpl.shape[0] > img_bin.shape[0] or tmpl.shape[1] > img_bin.shape[1]:
            continue
        res = cv2.matchTemplate(img_bin, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val >= threshold:
            return True, max_val  # Early exit to save time
        if max_val > best_score:
            best_score = max_val
    return False, best_score # return false if nothing found, best score has lower bound at 0



def match_symbol(image: np.ndarray,
                 template: np.ndarray,
                 symbol_contour: np.ndarray,
                 rotation_range=range(0, 75, 15),
                 shear_range=range(0,5,2),
                 log_to_csv: bool = False,
                 csv_path: str = "match_log.csv"):
    """
    Composite symbol matching: Template match + contour-based score. Optionally logs details to CSV.


    Args:
        image (np.ndarray): Unprocessed input image.
        template (np.ndarray): np.ndarray of templates/structural elements/masks.
        symbol_contour (np.ndarray): contour of the symbol being matched
        rotation_range (iterable)
        shear_range (iterable)
        log_to_csv (bool): If True, writes each match score to CSV. Exists for tuning the function.
        csv_path (str): File path for CSV logging.

    Returns:
        match_found (bool):
        matched_indices (list): Indices of matching contours.
        best_template_score (float)
        best_composite_score (float)

    """
    template = (template == 255).astype(np.uint8)
    image_copy = image.copy()
    img_bin = preprocess_meme(image_copy) #binarizaed and smooth meme
    # contours of the meme
    # swastika filter parameters
    # Swastika mask has 20 vertices and an aspect ratio of .99
    img_contours = find_filter_contours(img_bin,
                                           (0.9,1.36),
                                           (17,21),
                                           False,True)


    matched_indices = []
    template_scores = []
    composite_scores = []
    confirm_method = ""
    match_found = False

    # Compute dynamic scale_range so that
    scales = compute_scale_range(img_contours, symbol_contour)
    # regenerate templates at new scales, rotations, etc.
    dyn_templates = []

    for _, warped, rotation, scale, shear in transformation_generator(template, rotation_range, scales, shear_range):
        # warped are the regenerated symbols
        dyn_templates.append(warped)

    # rows for csv output
    # can't preallocate due to unknown number of contours
    rows = []

    # Morphological detection
    early_match = morphological_detect(img_bin, dyn_templates, img_contours)

    if early_match:
        print("Morphological match confirmed. Skipping further matching.")

        confirm_method = "morph"

        if log_to_csv:
            rows.append([[-1], early_match, -99, -99, -99, confirm_method])
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["index","match_found","template_score","match_score","composite_score","confirm_method"])
                writer.writerows(rows)
        return early_match, [-1], -99, -99, -99, confirm_method



    # Template matching
    template_found, template_score = match_template_with_transform(img_bin, dyn_templates, threshold=0.52)
    print(f"Template score of {template_score}")

    # If the template score is strong, trust it completely
    if template_found:
        confirm_method = "template"

        if log_to_csv:
            rows.append([[-1], template_found, template_score, -99, -99, confirm_method])
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["index", "match_found", "template_score","match_score", "composite_score", "confirm_method"])
                writer.writerows(rows)
        return template_found, [-1], template_score, -99, -99, confirm_method

    if not img_contours and not template_found:
        print("No contours found")
        # if no contours found return a
        # matched_indices, best_template_score, best_composite_score of -99
        return template_found, [-99], -99, -99, -99, confirm_method

    # weights for composite: morph=0.4, template=0.3, shape=0.3
    w_temp, w_shape = 0.5, 0.5

    for idx, cnt in enumerate(img_contours):
        # meme contour match score to symbol
        shape_score = cv2.matchShapes(symbol_contour, cnt, cv2.CONTOURS_MATCH_I3, 0.0)

        if shape_score < 0.1:
            match_found = True
            confirm_method = "shape"

            if log_to_csv:
                rows.append([idx, match_found, template_score, shape_score, -99, confirm_method])
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        ["index", "match_found", "template_score","match_score", "composite_score", "confirm_method"])
                    writer.writerows(rows)
            return True, [idx], template_score, shape_score, -99, confirm_method

        composite_score = w_temp*(1-template_score) + w_shape*shape_score

        if composite_score < 0.15:
            match_found = True
            print(f"Symbol matched at contour #{idx}!")
            matched_indices.append(idx)
            confirm_method = "composite"

        composite_scores.append(composite_score)

        if log_to_csv:
            rows.append([idx, match_found,  template_score, shape_score, composite_score, confirm_method])

    if log_to_csv:
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["index", "match_found", "template_score","match_score", "composite_score", "confirm_method"])
            writer.writerows(rows)
    # Template scores are from 0 to 1 with higher scores are better performing
    # Composite scores are unbounded and lower scores are better performing

    best_composite_score = min(composite_scores)
    return match_found, matched_indices, template_score, shape_score, best_composite_score, confirm_method

# Structural element functions

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

        se_templates[key] = se_bin
    return se_templates

