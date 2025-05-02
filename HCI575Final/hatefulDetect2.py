import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path


def display_image(image, title='Image'):
    """Display an image using matplotlib for better visualization."""
    plt.figure(figsize=(10, 8))
    if len(image.shape) == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def display_multiple(images, titles=None, figsize=(15, 10)):
    """Display multiple images in a row for comparison."""
    n = len(images)
    if titles is None:
        titles = [f'Image {i + 1}' for i in range(n)]

    plt.figure(figsize=figsize)
    for i in range(n):
        plt.subplot(1, n, i + 1)
        if len(images[i].shape) == 3:
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def create_template_variations():
    """
    Create multiple swastika template variations including filled and line-based versions.
    """
    templates = []

    # Various sizes for scale invariance
    for size in [24, 32, 40, 48]:
        # --- LINE-BASED TEMPLATES ---
        # Traditional swastika with arms facing right
        template_right = np.zeros((size, size), dtype=np.uint8)
        center = size // 2
        arm_length = size // 4
        line_width = max(1, size // 16)

        # Create the core cross
        cv2.line(template_right, (center - arm_length, center), (center + arm_length, center), 255, line_width)
        cv2.line(template_right, (center, center - arm_length), (center, center + arm_length), 255, line_width)

        # Add the hooks (clockwise)
        cv2.line(template_right, (center - arm_length, center), (center - arm_length, center - arm_length), 255,
                 line_width)
        cv2.line(template_right, (center, center - arm_length), (center + arm_length, center - arm_length), 255,
                 line_width)
        cv2.line(template_right, (center + arm_length, center), (center + arm_length, center + arm_length), 255,
                 line_width)
        cv2.line(template_right, (center, center + arm_length), (center - arm_length, center + arm_length), 255,
                 line_width)

        # Add template and its rotations (0, 45, 90, 135 degrees)
        for angle in range(0, 180, 45):
            M = cv2.getRotationMatrix2D((center, center), angle, 1.0)
            rotated = cv2.warpAffine(template_right, M, (size, size))
            templates.append(rotated)

        # Traditional swastika with arms facing left (mirror of right)
        template_left = np.zeros((size, size), dtype=np.uint8)
        # Core cross
        cv2.line(template_left, (center - arm_length, center), (center + arm_length, center), 255, line_width)
        cv2.line(template_left, (center, center - arm_length), (center, center + arm_length), 255, line_width)

        # Add the hooks (counter-clockwise)
        cv2.line(template_left, (center - arm_length, center), (center - arm_length, center + arm_length), 255,
                 line_width)
        cv2.line(template_left, (center, center - arm_length), (center - arm_length, center - arm_length), 255,
                 line_width)
        cv2.line(template_left, (center + arm_length, center), (center + arm_length, center - arm_length), 255,
                 line_width)
        cv2.line(template_left, (center, center + arm_length), (center + arm_length, center + arm_length), 255,
                 line_width)

        # Add template and its rotations
        for angle in range(0, 180, 45):
            M = cv2.getRotationMatrix2D((center, center), angle, 1.0)
            rotated = cv2.warpAffine(template_left, M, (size, size))
            templates.append(rotated)

        # --- FILLED TEMPLATES ---
        # Create filled versions of both right and left-facing swastikas
        for template in [template_right, template_left]:
            # Dilate to make thicker
            kernel = np.ones((3, 3), np.uint8)
            thick_template = cv2.dilate(template, kernel, iterations=1)

            # Add rotations of thick template
            for angle in range(0, 180, 45):
                M = cv2.getRotationMatrix2D((center, center), angle, 1.0)
                rotated = cv2.warpAffine(thick_template, M, (size, size))
                templates.append(rotated)

    # Add simplified swastika (just an "+" with small hooks)
    for size in [20, 28, 36]:
        simple_template = np.zeros((size, size), dtype=np.uint8)
        center = size // 2
        arm_length = size // 4
        hook_length = size // 8
        line_width = max(1, size // 20)

        # Draw the cross
        cv2.line(simple_template, (center - arm_length, center), (center + arm_length, center), 255, line_width)
        cv2.line(simple_template, (center, center - arm_length), (center, center + arm_length), 255, line_width)

        # Draw small hooks on ends
        cv2.line(simple_template, (center - arm_length, center), (center - arm_length, center - hook_length), 255,
                 line_width)
        cv2.line(simple_template, (center + arm_length, center), (center + arm_length, center + hook_length), 255,
                 line_width)
        cv2.line(simple_template, (center, center - arm_length), (center + hook_length, center - arm_length), 255,
                 line_width)
        cv2.line(simple_template, (center, center + arm_length), (center - hook_length, center + arm_length), 255,
                 line_width)

        for angle in range(0, 360, 45):
            M = cv2.getRotationMatrix2D((center, center), angle, 1.0)
            rotated = cv2.warpAffine(simple_template, M, (size, size))
            templates.append(rotated)

    return templates


def preprocess_image(image):
    """
    Enhanced preprocessing to highlight potential hateful symbols.
    Uses multiple methods and combines results.
    """
    # Ensure we have a BGR image to start with
    if len(image.shape) == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = image.copy()

    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Method 1: Edge detection with Canny
    edges = cv2.Canny(gray, 50, 150)

    # Method 2: Adaptive thresholding to find lines and symbols
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Method 3: Color-based segmentation for dark lines on light backgrounds
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # Target dark colors (black/dark gray)
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 90])
    dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)

    # Combine the methods
    combined = cv2.bitwise_or(edges, thresh)
    combined = cv2.bitwise_or(combined, dark_mask)

    # Clean up noise
    kernel = np.ones((2, 2), np.uint8)
    denoised = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

    # Enhance the remaining features
    enhanced = cv2.dilate(denoised, kernel, iterations=1)

    return enhanced


def detect_hate_symbols(image, templates, min_threshold=0.50, debug=False):
    """
    Improved detection function that combines template matching with geometric verification.

    Args:
        image: Input image (BGR format)
        templates: List of template images
        min_threshold: Minimum correlation threshold
        debug: Whether to show debug info

    Returns:
        tuple: (image with detections, list of detections)
    """
    if image is None:
        return None, []

    # Create a copy of the input for visualization
    result_img = image.copy()

    # Preprocess the image to highlight potential symbols
    processed = preprocess_image(image)

    # For visualization if debug is True
    debug_images = [image, processed]
    debug_titles = ['Original', 'Processed']

    # Store detections
    detections = []

    # Apply template matching for each template
    for idx, template in enumerate(templates):
        # Skip templates that are too large for the image
        if template.shape[0] > processed.shape[0] or template.shape[1] > processed.shape[1]:
            continue

        # Apply template matching
        result = cv2.matchTemplate(processed, template, cv2.TM_CCOEFF_NORMED)

        # Find positions where the matching exceeds threshold
        locations = np.where(result >= min_threshold)

        # For each detection
        for pt in zip(*locations[::-1]):  # Reverse locations to get (x, y)
            w, h = template.shape[::-1]
            score = result[pt[1], pt[0]]

            # Extract the region for geometric verification
            roi = processed[pt[1]:pt[1] + h, pt[0]:pt[0] + w]

            # Skip if ROI is empty or invalid
            if roi.size == 0 or roi.shape[0] != h or roi.shape[1] != w:
                continue

            # Calculate feature overlap between template and ROI
            # (This helps reduce false positives)
            template_pixels = np.sum(template > 0)
            if template_pixels == 0:  # Avoid division by zero
                continue

            # Count overlapping white pixels
            overlap = np.logical_and(roi > 0, template > 0)
            overlap_count = np.sum(overlap)

            # Calculate overlap ratio
            overlap_ratio = overlap_count / template_pixels

            # Only keep detections with good overlap
            if overlap_ratio < 0.3:  # Adjust this threshold as needed
                continue

            # Check if this detection overlaps with previous ones
            new_detection = True
            for d in detections:
                x1, y1, w1, h1 = d['location']

                # Calculate overlap between this detection and existing one
                overlap_x = max(0, min(pt[0] + w, x1 + w1) - max(pt[0], x1))
                overlap_y = max(0, min(pt[1] + h, y1 + h1) - max(pt[1], y1))
                overlap_area = overlap_x * overlap_y

                # If there's significant overlap
                if overlap_area > (w * h * 0.5) or overlap_area > (w1 * h1 * 0.5):
                    new_detection = False
                    # Keep the detection with higher score
                    if score > d['score']:
                        d['score'] = score
                        d['location'] = (pt[0], pt[1], w, h)
                        d['template_idx'] = idx
                        d['overlap_ratio'] = overlap_ratio
                    break

            if new_detection:
                detections.append({
                    'location': (pt[0], pt[1], w, h),
                    'score': score,
                    'template_idx': idx,
                    'overlap_ratio': overlap_ratio
                })

    # Sort detections by score
    detections = sorted(detections, key=lambda x: x['score'], reverse=True)

    # Draw rectangles for each detection
    for d in detections:
        x, y, w, h = d['location']
        score = d['score']

        # Use red for high confidence, yellow for medium
        if score > 0.6:
            color = (0, 0, 255)  # Red (BGR)
        else:
            color = (0, 165, 255)  # Orange (BGR)

        cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(result_img, f"{score:.2f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Show debug visualization if requested
    if debug and detections:
        debug_images.append(result_img)
        debug_titles.append('Detections')
        display_multiple(debug_images, debug_titles)

    return result_img, detections


def process_batch(input_folder, output_folder=None, debug=False):
    """
    Process multiple memes in batch mode.

    Args:
        input_folder: Folder containing meme images
        output_folder: Folder to save results (if None, don't save)
        debug: Whether to show debug visualizations

    Returns:
        dict: Statistics about the processing
    """
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create templates only once
    print("Creating template variations...")
    templates = create_template_variations()
    print(f"Created {len(templates)} template variations")

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(Path(input_folder).glob(f'*{ext}')))

    print(f"Found {len(image_files)} images to process")

    # Process each image
    results = {
        'total': len(image_files),
        'with_detections': 0,
        'all_detections': []
    }

    for i, img_path in enumerate(image_files):
        print(f"Processing image {i + 1}/{len(image_files)}: {img_path.name}")

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  Error: Could not load image {img_path}")
            continue

        # Detect hate symbols
        result_img, detections = detect_hate_symbols(image, templates, debug=debug)

        # Record results
        if detections:
            results['with_detections'] += 1
            results['all_detections'].append({
                'file': img_path.name,
                'count': len(detections),
                'scores': [d['score'] for d in detections]
            })
            print(f"  Found {len(detections)} potential hateful symbols")
        else:
            print("  No hateful symbols detected")

        # Save result if output folder is provided
        if output_folder:
            out_path = os.path.join(output_folder, img_path.name)
            cv2.imwrite(out_path, result_img)

    print("\nBatch processing complete!")
    print(f"Processed {results['total']} images")
    print(f"Found potential hateful symbols in {results['with_detections']} images")

    return results


def test_single_image(image_path, debug=True):
    """Test detection on a single image and display results."""
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    # Create templates
    templates = create_template_variations()

    # Detect hate symbols
    result_img, detections = detect_hate_symbols(image, templates, debug=debug)

    # Print results
    print(f"Found {len(detections)} potential hateful symbols")
    for i, d in enumerate(detections):
        print(f"  Detection {i + 1}: Score {d['score']:.2f} at position {d['location'][:2]}")

    # Display results if not in debug mode (debug mode already displays)
    if not debug and detections:
        display_multiple([image, result_img], ['Original', 'Detections'])

    return result_img, detections

# Example usage for single image
test_single_image('images/0000001.png')

# Example usage for batch processing
# process_batch('input_folder', 'output_folder', debug=False)