import os
import sys
import csv
from PIL import Image



def process_images(input_folder, prefix, output_folder=None, csv_filename="image_list.csv"):
    """
    Process all images in a folder:
    1. Rename them to {prefix}_XXXX format
    2. Convert them to PNG format
    3. Create a CSV with all the image names

    Args:
        input_folder (str): Path to the folder containing images
        prefix (str): Prefix to use for renamed images
        output_folder (str, optional): Path to save the processed images. If None, uses input_folder
        csv_filename (str, optional): Name of the CSV file to create
    """
    # If no output folder specified, use the input folder
    if output_folder is None:
        output_folder = input_folder

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get all files in the input folder
    files = os.listdir(input_folder)

    # Filter for image files (basic filtering by extension)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
    image_files = [f for f in files if os.path.splitext(f.lower())[1] in image_extensions]

    if not image_files:
        print("No image files found in the specified folder.")
        return

    # List to store the new image names
    new_image_names = []

    # Process each image
    for i, image_file in enumerate(image_files, 1):
        try:
            # Open the image
            img_path = os.path.join(input_folder, image_file)
            img = Image.open(img_path)

            # Convert the image to RGB mode if it has an alpha channel
            if img.mode == 'RGBA':
                background = Image.new('RGBA', img.size, (255, 255, 255))
                img = Image.alpha_composite(background, img).convert('RGB')
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            # Create new name with format {prefix}_XXXX
            new_name = f"{prefix}_{i:04d}.png"
            new_path = os.path.join(output_folder, new_name)

            # Convert and save as PNG without color profile to avoid iCCP warnings
            img.save(new_path, 'PNG', icc_profile=None)

            # Add to our list of names
            new_image_names.append([new_name])

            print(f"Processed: {image_file} -> {new_name}")

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    # Write the CSV file
    csv_path = os.path.join(output_folder, csv_filename)
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)  # Using the renamed module
        # Write header
        writer.writerow(['image_name'])
        # Write image names
        writer.writerows(new_image_names)

    print(f"\nProcessed {len(new_image_names)} images.")
    print(f"CSV file created at: {csv_path}")


if __name__ == "__main__":
    input_folder = "images_og"
    prefix = "hateful"
    output = "images"
    csv_filename = "labels.csv"
    process_images(input_folder, prefix, output, csv_filename)