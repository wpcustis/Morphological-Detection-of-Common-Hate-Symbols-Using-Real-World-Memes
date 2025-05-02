import cv2
import numpy as np


def build_transformation_matrix(center, rotation_deg, scale, shear_deg):
    """
    Build a composite transformation matrix (3x3) that applies scaling, rotation, and shear about the center.

    Parameters:
      center (tuple): (x, y) center of the image.
      rotation_deg (float): Rotation angle in degrees.
      scale (float): Scaling factor.
      shear_deg (float): Shear angle in degrees (applied along x axis).

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


def transformation_generator(image, rotation_range, scale_range, shear_range):
    """
    Generator that yields affine transformation matrices and the resulting warped images.

    Parameters:
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



if __name__ == '__main__':
    # Load an example image.
    image = cv2.imread('Swastika_1.webp')
    if image is None:
        raise ValueError("Image not found. Please check the file path.")

    # Define parameter ranges.
    rotation_range = range(0, 85, 5)  # 0 to 85 degrees in increments of 5.
    scale_range = np.linspace(0.5, 1.2, num=4)  # Adjust these values as needed.
    shear_range = np.linspace(-10, 10, num=5)  # Shear angles from -10 to 10 degrees.

    # Iterate over transformations.
    for idx, (M, warped, rot, scale, shear) in enumerate(
            transformation_generator(image, rotation_range, scale_range, shear_range)):
        # Display each warped image in a window.
        cv2.imshow(f'Warped_{idx}', warped)
        # Optionally, save the warped image.
        # cv2.imwrite(f'warped_{idx}.jpg', warped)
        print(f"Transformation {idx}: Rotation={rot}, Scale={scale:.2f}, Shear={shear:.2f}")

        # Wait for key press (press any key to close the window and continue).
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Optionally, close all windows at the end.
    cv2.destroyAllWindows()
