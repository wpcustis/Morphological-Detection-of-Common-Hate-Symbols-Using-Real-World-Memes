import cv2
import numpy as np
import os
from contour3 import needs_smoothing
from contour3 import smooth_image
from contour3 import load_structural_elements

def enhanced_preprocess_meme(image: np.ndarray) -> np.ndarray:
    """Enhanced preprocessing pipeline with more robust handling of varying meme qualities"""
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_eq = clahe.apply(img)

    # Determine if smoothing is needed
    if needs_smoothing(img):
        img_eq = smooth_image(img_eq)

    # Create multiple binarization methods and combine results
    # Otsu's method
    _, otsu_bin = cv2.threshold(img_eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Adaptive thresholding
    adaptive_bin = cv2.adaptiveThreshold(img_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)

    # Combine the two methods for more robust detection
    combined_bin = cv2.bitwise_or(otsu_bin, adaptive_bin)

    # Apply morphological operations to clean up the binary image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned_bin = cv2.morphologyEx(combined_bin, cv2.MORPH_OPEN, kernel, iterations=1)

    return cleaned_bin

def extract_features(image: np.ndarray):
    """
    Extract multiple feature types for more robust symbol detection.
    Includes SIFT features, HOG descriptors, and shape context.

    Args:
        image (np.ndarray): Input grayscale image

    Returns:
        dict: Dictionary of extracted features
    """
    features = {}

    # SIFT features for keypoint matching
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    features['sift'] = {'keypoints': keypoints, 'descriptors': descriptors}

    # # HOG features for shape description
    # win_size = (64, 64)
    # if image.shape[0] >= win_size[0] and image.shape[1] >= win_size[1]:
    #     # Resize image to fit HOG window if needed
    #     resized = cv2.resize(image, win_size) if image.shape[:2] != win_size else image
    #
    #     # Configure HOG
    #     win_stride = (8, 8)
    #     padding = (8, 8)
    #     hog = cv2.HOGDescriptor()
    #
    #     # Convert to grayscale if image is in color
    #     if len(resized.shape) > 2 and resized.shape[2] > 1:
    #         resized_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    #     else:
    #         resized_gray = resized
    #
    #     # Compute HOG features
    #     features['hog'] = hog.compute(
    #         resized_gray,
    #         winStride=win_stride,
    #         padding=padding
    #     )
    # else:
    #     features['hog'] = None

    # Shape context through contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # Compute Hu moments for shape description
        moments = cv2.moments(largest_contour)
        hu_moments = cv2.HuMoments(moments)
        features['hu_moments'] = hu_moments
    else:
        features['hu_moments'] = None

    return features


def detect_symbol_multiscale(image: np.ndarray,
                             template: np.ndarray,
                             min_scale: float = 0.1,
                             max_scale: float = 2.0,
                             scale_steps: int = 10,
                             rotation_steps: int = 12,
                             threshold: float = 0.65):
    """
    Multi-scale and multi-orientation symbol detection in images.

    Args:
        image (np.ndarray): Input image
        template (np.ndarray): Template image of the symbol
        min_scale (float): Minimum scale factor
        max_scale (float): Maximum scale factor
        scale_steps (int): Number of scale steps to check
        rotation_steps (int): Number of rotation angles to check
        threshold (float): Detection threshold (0-1)

    Returns:
        list: List of detections with format [(x, y, w, h, confidence, angle, scale), ...]
    """
    # Preprocess the image
    preprocessed = enhanced_preprocess_meme(image)

    # Preprocess the template
    if len(template.shape) == 3:
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        template_gray = template

    _, template_bin = cv2.threshold(template_gray, 127, 255, cv2.THRESH_BINARY)

    # Generate scales
    scales = np.linspace(min_scale, max_scale, scale_steps)

    # Generate rotation angles (0-360 degrees)
    angles = np.linspace(0, 360, rotation_steps, endpoint=False)

    detections = []

    # Extract template features just once
    template_features = extract_features(template_bin)

    h, w = template_bin.shape
    for scale in scales:
        # Calculate new dimensions
        new_w, new_h = int(w * scale), int(h * scale)

        # Skip if scaled template is too small
        if new_w < 8 or new_h < 8:
            continue

        # Scale the template
        scaled_template = cv2.resize(template_bin, (new_w, new_h))

        for angle in angles:
            # Skip angles that would result in same appearance for symmetric symbols
            # For example, swastika has 90-degree rotational symmetry
            # This can be customized based on the specific symbol
            if angle > 0 and (angle % 90 == 0) and 'swastika' in str(template):
                continue

            # Apply rotation to the template
            rotation_matrix = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), angle, 1)
            rotated_template = cv2.warpAffine(scaled_template, rotation_matrix, (new_w, new_h))

            # Create a larger frame to hold rotated template
            # (rotations can cut off corners if not in a larger frame)
            diagonal = int(np.sqrt(new_w ** 2 + new_h ** 2)) + 2
            padded_template = np.zeros((diagonal, diagonal), dtype=np.uint8)
            x_offset = (diagonal - new_w) // 2
            y_offset = (diagonal - new_h) // 2
            padded_template[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = rotated_template

            # Match template in the image
            result = cv2.matchTemplate(preprocessed, padded_template, cv2.TM_CCOEFF_NORMED)

            # Extract candidate locations
            locations = np.where(result >= threshold)

            for pt in zip(*locations[::-1]):  # Switch x and y coordinates
                # Get the patch from the image for feature matching
                patch = preprocessed[pt[1]:pt[1] + diagonal, pt[0]:pt[0] + diagonal]

                # Skip if patch is not the right size (edge cases)
                if patch.shape[0] != diagonal or patch.shape[1] != diagonal:
                    continue

                # Extract features from the patch
                patch_features = extract_features(patch)

                # Match features for a more robust verification
                feature_match_score = match_features(patch_features, template_features)

                # Template match score
                template_score = result[pt[1], pt[0]]

                # Combined score
                combined_score = 0.7 * template_score + 0.3 * feature_match_score

                if combined_score >= threshold:
                    detections.append({
                        'x': pt[0],
                        'y': pt[1],
                        'width': diagonal,
                        'height': diagonal,
                        'confidence': combined_score,
                        'angle': angle,
                        'scale': scale
                    })

    # Apply non-maximum suppression to remove overlapping detections
    filtered_detections = non_max_suppression(detections, 0.3)

    return filtered_detections


def non_max_suppression(detections, overlap_threshold=0.3):
    """
    Apply non-maximum suppression to remove overlapping detections.

    Args:
        detections (list): List of detection dictionaries
        overlap_threshold (float): Maximum allowed overlap (IoU)

    Returns:
        list: Filtered list of detections
    """
    if not detections:
        return []

    # Sort by confidence
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

    # Output list
    filtered = []

    while detections:
        # Take the detection with highest confidence
        best = detections.pop(0)
        filtered.append(best)

        # Remove detections that overlap too much with the best one
        remaining = []
        for det in detections:
            # Calculate IoU
            # Convert detection format to [x1, y1, x2, y2]
            box1 = [best['x'], best['y'], best['x'] + best['width'], best['y'] + best['height']]
            box2 = [det['x'], det['y'], det['x'] + det['width'], det['y'] + det['height']]

            # Calculate intersection
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])

            intersection = max(0, x2 - x1) * max(0, y2 - y1)

            # Calculate areas
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

            # Calculate IoU
            iou = intersection / float(area1 + area2 - intersection)

            if iou < overlap_threshold:
                remaining.append(det)

        detections = remaining

    return filtered


def augment_template(template: np.ndarray, num_variants: int = 10):
    """
    Create augmented variants of a template to handle style variations.
    Particularly useful for matching hand-drawn versions of symbols.

    Args:
        template (np.ndarray): Original template image
        num_variants (int): Number of variants to generate

    Returns:
        list: List of augmented template images
    """
    augmented_templates = [template]  # Include the original

    for i in range(num_variants):
        variant = template.copy()

        # Apply random transformations

        # 1. Random dilation or erosion to simulate line thickness variations
        kernel_size = np.random.randint(1, 4) * 2 + 1  # Odd kernel sizes: 3, 5, 7
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        if np.random.random() < 0.5:
            # Dilate
            variant = cv2.dilate(variant, kernel, iterations=1)
        else:
            # Erode
            variant = cv2.erode(variant, kernel, iterations=1)

        # 2. Add random perspective distortion to simulate viewing angle changes
        h, w = variant.shape[:2]
        src_pts = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])

        # Create random destination points with small perturbations
        max_perturbation = min(w, h) * 0.2
        dst_pts = src_pts + np.random.uniform(-max_perturbation, max_perturbation, src_pts.shape)

        # Ensure points stay within reasonable bounds
        dst_pts = np.clip(dst_pts, [0, 0], [w - 1, h - 1])

        # Apply perspective transform
        M = cv2.getPerspectiveTransform(src_pts.astype(np.float32), dst_pts.astype(np.float32))
        variant = cv2.warpPerspective(variant, M, (w, h))

        # 3. Add some noise to simulate hand-drawing imperfections
        if np.random.random() < 0.7:  # 70% chance to add noise
            noise = np.random.normal(0, 15, variant.shape).astype(np.int32)
            variant = np.clip(variant.astype(np.int32) + noise, 0, 255).astype(np.uint8)

            # Apply slight Gaussian blur to smooth out noise
            variant = cv2.GaussianBlur(variant, (3, 3), 0)

            # Re-threshold to ensure binary image
            _, variant = cv2.threshold(variant, 127, 255, cv2.THRESH_BINARY)

        # 4. Random elastic deformations for hand-drawn effect
        if np.random.random() < 0.5:  # 50% chance for elastic deformations
            # Create displacement maps
            dx = np.random.uniform(-5, 5, variant.shape).astype(np.float32)
            dy = np.random.uniform(-5, 5, variant.shape).astype(np.float32)

            # Smooth the displacement maps
            dx = cv2.GaussianBlur(dx, (31, 31), 11)
            dy = cv2.GaussianBlur(dy, (31, 31), 11)

            # Create deformation maps
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = np.float32(x + dx)
            map_y = np.float32(y + dy)

            # Apply deformation
            variant = cv2.remap(variant, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            # Re-threshold after deformation
            _, variant = cv2.threshold(variant, 127, 255, cv2.THRESH_BINARY)

        augmented_templates.append(variant)

    return augmented_templates


def create_style_variants(template: np.ndarray):
    """
    Create specific style variants to match common representations.

    Args:
        template (np.ndarray): Original template image

    Returns:
        dict: Dictionary of styled variants
    """
    styles = {}
    styles['original'] = template

    # Digital/clean variant
    digital = template.copy()
    # Ensure clean edges
    digital = cv2.dilate(digital, np.ones((3, 3), np.uint8), iterations=1)
    digital = cv2.erode(digital, np.ones((3, 3), np.uint8), iterations=1)
    styles['digital'] = digital

    # Hand-drawn thick variant
    hand_drawn_thick = template.copy()
    hand_drawn_thick = cv2.dilate(hand_drawn_thick, np.ones((5, 5), np.uint8), iterations=2)
    # Add some irregularity to edges
    kernel = np.ones((3, 3), np.uint8)
    hand_drawn_thick = cv2.erode(hand_drawn_thick, kernel, iterations=1)

    # Add noise to edges
    contours, _ = cv2.findContours(hand_drawn_thick, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    edge_mask = np.zeros_like(hand_drawn_thick)
    cv2.drawContours(edge_mask, contours, -1, 255, 2)

    # Apply noise only to edges
    noise = np.random.randint(0, 2, edge_mask.shape, np.uint8) * 255
    noise_on_edges = cv2.bitwise_and(noise, edge_mask)
    hand_drawn_thick = cv2.bitwise_xor(hand_drawn_thick, noise_on_edges)

    styles['hand_drawn_thick'] = hand_drawn_thick

    # Hand-drawn thin variant
    hand_drawn_thin = template.copy()
    # Make lines thinner
    hand_drawn_thin = cv2.erode(hand_drawn_thin, np.ones((3, 3), np.uint8), iterations=1)
    # Add slight distortion
    rows, cols = hand_drawn_thin.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 0, 1.05)
    hand_drawn_thin = cv2.warpAffine(hand_drawn_thin, M, (cols, rows))

    # Add small breaks in lines to simulate hand-drawing
    kernel = np.ones((2, 2), np.uint8)
    hand_drawn_thin = cv2.morphologyEx(hand_drawn_thin, cv2.MORPH_OPEN, kernel)

    styles['hand_drawn_thin'] = hand_drawn_thin

    # Sketchy variant
    sketchy = template.copy()
    # Create a parallel line effect
    kernel = np.ones((1, 3), np.uint8)  # Horizontal kernel
    sketch_h = cv2.erode(sketchy, kernel, iterations=1)

    kernel = np.ones((3, 1), np.uint8)  # Vertical kernel
    sketch_v = cv2.erode(sketchy, kernel, iterations=1)

    # Combine horizontal and vertical erosion
    sketchy = cv2.bitwise_or(sketch_h, sketch_v)

    # Add some noise
    noise = np.random.normal(0, 25, sketchy.shape).astype(np.int32)
    sketchy = np.clip(sketchy.astype(np.int32) + noise, 0, 255).astype(np.uint8)
    _, sketchy = cv2.threshold(sketchy, 127, 255, cv2.THRESH_BINARY)

    styles['sketchy'] = sketchy

    return styles


def match_features(query_features, template_features, thresholds={'sift': 0.7, 'hu': 0.3}): # 'hog': 0.8,
    """
    Match features using multiple methods and combine scores.

    Args:
        query_features (dict): Features extracted from query image
        template_features (dict): Features extracted from template
        thresholds (dict): Thresholds for different feature types

    Returns:
        float: Composite match score (0-1, higher is better match)
    """
    scores = []

    # SIFT matching
    if query_features['sift']['descriptors'] is not None and template_features['sift']['descriptors'] is not None:
        # Use FLANN matcher for faster matching
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(
            query_features['sift']['descriptors'],
            template_features['sift']['descriptors'],
            k=2
        )

        # Apply Lowe's ratio test to filter good matches
        good_matches = []
        for m, n in matches:
            if m.distance < thresholds['sift'] * n.distance:
                good_matches.append(m)

        sift_score = len(good_matches) / max(len(matches), 1)
        scores.append(sift_score)

    # # HOG similarity
    # if query_features['hog'] is not None and template_features['hog'] is not None:
    #     # Compute cosine similarity between HOG descriptors
    #     query_hog = query_features['hog'].flatten()
    #     template_hog = template_features['hog'].flatten()
    #
    #     dot_product = np.dot(query_hog, template_hog)
    #     norm_product = np.linalg.norm(query_hog) * np.linalg.norm(template_hog)
    #
    #     if norm_product > 0:
    #         hog_similarity = dot_product / norm_product
    #         if hog_similarity > thresholds['hog']:
    #             scores.append(hog_similarity)

    # Shape matching with Hu moments
    if query_features['hu_moments'] is not None and template_features['hu_moments'] is not None:
        # Compare Hu moments
        match_shape = cv2.matchShapes(
            query_features['hu_moments'],
            template_features['hu_moments'],
            cv2.CONTOURS_MATCH_I1,
            0
        )
        # Convert to similarity score (0-1)
        shape_similarity = max(0, 1 - match_shape)
        if shape_similarity > thresholds['hu']:
            scores.append(shape_similarity)

    # Combine scores
    if scores:
        return sum(scores) / len(scores)
    else:
        return 0.0

def detect_hate_symbols(image_path, templates_dir, output_path=None):
    """
    Complete hate symbol detection pipeline.

    Args:
        image_path (str): Path to the input image/meme
        templates_dir (str): Directory containing template images
        output_path (str): Path to save visualization (optional)


    Returns:
        detections (list): List of detected symbol instances
        annotated_image (np.ndarray): Image with annotations
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Load templates
    templates = load_structural_elements(templates_dir)

    # Create augmented template variants
    augmented_templates = {}
    for symbol_name, template in templates.items():
        # Create standard style variants
        style_variants = create_style_variants(template)

        # Further augment each style
        augmented_templates[symbol_name] = {}
        for style_name, variant in style_variants.items():
            augmented_templates[symbol_name][style_name] = augment_template(variant, num_variants=5)

    # Detect symbols
    all_detections = []

    # Preprocess the input image once
    preprocessed = enhanced_preprocess_meme(image)

    # Run detection for each template and its variants
    for symbol_name, styles in augmented_templates.items():
        for style_name, variants in styles.items():
            for i, variant in enumerate(variants):
                # Multi-scale detection
                detections = detect_symbol_multiscale(
                    preprocessed,
                    variant,
                    min_scale=0.05,
                    max_scale=2.0,
                    scale_steps=8,
                    rotation_steps=8,
                    threshold=0.65
                )

                # Add metadata to detections
                for det in detections:
                    det['symbol'] = symbol_name
                    det['style'] = style_name
                    det['variant'] = i
                    all_detections.append(det)

    # Remove duplicates through non-maximum suppression
    final_detections = non_max_suppression(all_detections, 0.3)

    # Draw detections on the image
    annotated_image = draw_detections(image, final_detections)

    # Save output if requested
    if output_path:
        cv2.imwrite(output_path, annotated_image)

    return final_detections, annotated_image


def draw_detections(image, detections):
    """
    Draw detection boxes and information on the image.

    Args:
        image (np.ndarray): Input image
        detections (list): List of detection dictionaries

    Returns:
        np.ndarray: Annotated image
    """
    result = image.copy()

    # Define colors for different symbols
    colors = {
        'swastika': (0, 0, 255),  # Red (BGR format)
        'ss': (255, 0, 0)  # Blue
    }

    for det in detections:
        x, y, w, h = det['x'], det['y'], det['width'], det['height']
        symbol = det['symbol']
        confidence = det.get('final_confidence', det['confidence'])

        # Get color for this symbol
        color = colors.get(symbol, (0, 255, 0))  # Default to green

        # Draw rectangle
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

        # Draw label
        label = f"{symbol}: {confidence:.2f}"
        cv2.putText(result, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return result

if __name__ == "__main__":

    input_dir = "images"
    output_dir = "results"
    templates_dir = "se_dir"
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each image in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            print(f"Processing {filename}...")
            # try:
            detections, annotated = detect_hate_symbols(
                input_path,
                templates_dir,
                output_path
            )

            # Write detection results to text file
            txt_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_detections.txt")
            with open(txt_path, 'w') as f:
                for i, det in enumerate(detections, 1):
                    f.write(f"Detection {i}:\n")
                    f.write(f"  Symbol: {det['symbol']}\n")
                    f.write(f"  Style: {det['style']}\n")
                    f.write(f"  Position: ({det['x']}, {det['y']}, {det['width']}x{det['height']})\n")
                    f.write(f"  Confidence: {det.get('final_confidence', det['confidence']):.3f}\n")
                    f.write("\n")

            print(f"  Found {len(detections)} symbols")
            # except Exception as e:
            #     print(f"  Error processing {filename}: {e}")