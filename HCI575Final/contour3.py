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
    print(f"image Laplace score: {laplacian_var}")
    smooth_bool = laplacian_var > threshold * 1000
    return smooth_bool

def smooth_image(image):
    # Basic smoothing steps
    blurred = cv2.GaussianBlur(image, (5, 5), 0)  # Smooth noise
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
        print("Image smoothed.")
        # display_image(img)


    # Compute mean intensity to decide thresholding method due to variance in meme brightness
    mean_intensity = np.mean(img)
    print(f"Mean Intensity:{mean_intensity}")

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

    # display_image(img_bin)

    # # Morphological opening to remove small white speckles
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel, iterations=1)
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
                    aspect_range: tuple = (0.8, 1.2),
                    vertex_range: tuple = (3, 12),
                    convexity: bool = None,
                    remove_largest: bool = True,
                    approx_eps: float = 0.01) -> list:
    """
    Return only those contours that satisfy:
      - aspect_ratio = width / height within aspect_range
      - number of vertices within vertex_range
      - convexity matches
    In exploration for this project, these elements were the most important for both templates

    Args:
        image (np.ndarray): The image
        aspect_range (tuple): Contains (min_aspect, max_aspect)
        vertex_range (tuple): (min_vertices, max_vertices)
        convexity (bool): True to require convex, False to require concave,
                        or None to accept both.
        approx_eps (float): fraction of perimeter to use for polygonal approx. Default = 0.01.
        remove_largest (bool): removes largest area object in contours since this uses cv2.RETR_LIST
        to grab all contours, and one inevitably will be the contour for the whole image.
        Defaults to True.

    Returns:
        filtered (list): A filtered list of contours that conform to the desired specs.
    """
    image_copy = image.copy()
    image_copy = preprocess_meme(image_copy)
    contours, hierarchy = cv2.findContours(image_copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours_copy = list(contours)

    # Remove largest area contour
    if remove_largest:
        # find index of largest area contour
        largest_idx = max(range(len(contours_copy)), key=lambda i: cv2.contourArea(contours_copy[i]))
        # Remove the largest contour
        contours_copy.pop(largest_idx)


    filtered = []
    min_aspect, max_aspect = aspect_range
    min_verts, max_verts = vertex_range

    for cnt in contours_copy:
        # Compute bounding rect and aspect ratio
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
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

    return filtered

def compute_scale_range(contours: tuple,
                        symbol_contour: np.ndarray,
                        margin: float = 0.2,
                        absolute_min: float = 0.05,
                        absolute_max: float = 5.0,
                        steps: int = 5) -> list[float]:
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
        print("No valid contours found.")
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
    print(templates)
    for tmpl in templates:
        # ADD: ensure template not larger than image
        print(tmpl.shape)
        if tmpl.shape[0] > img_bin.shape[0] or tmpl.shape[1] > img_bin.shape[1]:
            continue
        res = cv2.matchTemplate(img_bin, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        best_score = max(best_score, max_val)

    match_found = best_score >= threshold
    return match_found, best_score

def match_symbol(img_bin: np.ndarray,
                 template: np.ndarray,
                 symbol_contour: np.ndarray,
                 rotation_range=range(0, 90, 15),
                 shear_range=range(0,5,1),
                 log_to_csv: bool = False,
                 csv_path: str = "match_log.csv") -> list:
    """
    Composite symbol matching: Template match + contour-based score. Optionally logs details to CSV.


    Args:
        img_bin (np.ndarray): Binarized and processed input image.
        templates (list): List of templates/structural elements.
        symbol_contour (np.ndarray): contour of the symbol being matched
        rotation_range (iterable)
        shear_range (iterable)
        log_to_csv (bool): If True, writes each match score to CSV. Exists for tuning the function.
        csv_path (str): File path for CSV logging.

    Returns:
        matched_indices (list): Indices of matching contours.

    """
    # contours of the meme
    img_contours, _ = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)



    matched_indices = []

    # Symbol vertices
    symbol_perimeter = cv2.arcLength(symbol_contour, True)
    symbol_approx = cv2.approxPolyDP(symbol_contour, 0.01 * symbol_perimeter, True)
    symbol_vertex_count = len(symbol_approx)

    # Symbol aspect ratio
    sym_x, sym_y, sym_w, sym_h = cv2.boundingRect(symbol_contour)
    symbol_aspect_ratio = sym_w / sym_h if sym_h != 0 else 0

    # ADD: Compute dynamic scale_range if needed
    scales = compute_scale_range(img_contours, symbol_contour)

    # regenerate templates at new scales
    dyn_templates = []

    for _, warped, rot, scale, shear in transformation_generator(template, rotation_range, scales, shear_range):
        # ADD: include warped only, ignoring transform details
        dyn_templates.append(warped)
    # template matching
    match_found, template_score = match_template_with_transform(img_bin, dyn_templates, threshold=0.7)
    print(f"Template score of {template_score}")

    # rows for csv output
    # can't preallocate due to unknown number of contours
    rows = []

    for idx, cnt in enumerate(img_contours):
        # meme contour match score to symbol
        match_score = cv2.matchShapes(symbol_contour, cnt, cv2.CONTOURS_MATCH_I1, 0.0)

        # meme contours vertices
        cnt_perimeter = cv2.arcLength(cnt, True)
        cnt_approx = cv2.approxPolyDP(cnt, 0.01 * cnt_perimeter, True)
        vertex_diff = abs(len(cnt_approx) - symbol_vertex_count)
        vertex_score = vertex_diff / symbol_vertex_count
        # meme contour aspect ratio
        x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
        cnt_aspect_ratio = w_cnt / h_cnt if h_cnt != 0 else 0
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

    return matched_indices, template_score, composite_score

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


# ADD: Fix test call to use correct function
if __name__ == '__main__':
    test_templates = load_structural_elements("se_dir")
    swastika_se = test_templates.get('swastika', [])

    # contours of the template
    # these were saved since it made no sense to me to constantly be reloading them and recomputing them
    # see helper/templateContours.py for more information

    swastika_contour = np.load('contours/swastika_contour.npy')
    ss_contour = np.load('contours/ss_contour.npy')
    testmeme1 = cv2.imread('images/0000001.png')
    testMeme = preprocess_meme('images/0000001.png')
    draw_contours, _ = cv2.findContours(testMeme,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    drawncontour = cv2.drawContours(testmeme1,draw_contours,101, (100,255,100),2)

    display_image(drawncontour)

    exit()
    # ADD: call match_template_with_transform correctly
    found, score = match_symbol(testMeme, swastika_se,swastika_contour)
    print(f"Template match result: found={found}, score={score}")
