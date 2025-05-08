import cv2
import numpy as np
import os
from detectionFunctions import preprocess_meme

# preprocesses all memes to get a top-down view of issues after multiple tests and methods returned back poor results

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            image = cv2.imread(input_path)

            if image is None:
                print(f"Failed to load image: {input_path}")
                continue

            processed_img = preprocess_meme(image)

            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, processed_img)
            print(f"Processed and saved: {output_path}")

# Example usage
if __name__ == "__main__":
    input_dir = "images"
    output_dir = "bin_images"
    process_folder(input_dir, output_dir)
