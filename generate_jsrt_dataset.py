import os
import argparse
import numpy as np
import cv2

# Argument parsing
parser = argparse.ArgumentParser(description="Preprocess JSRT images for unsupervised learning")
parser.add_argument("--data_dir", type=str, default=r'C:\Users\tejes\Downloads\All247images (2)\All247images\images')
parser.add_argument("--result_dir", type=str, default=r'C:\Users\tejes\Downloads\All247images (2)\All247images\converted_jsrt')
args = parser.parse_args()

data_dir = args.data_dir
result_dir = args.result_dir

# Create result directory for images
image_result_dir = os.path.join(result_dir, 'image')
if not os.path.exists(image_result_dir):
    os.makedirs(image_result_dir)

# Process each .IMG file
files = os.listdir(data_dir)
for file in files:
    if not file.endswith('.IMG'):
        continue

    # Read the raw image data
    with open(os.path.join(data_dir, file), 'rb') as fid:
        dtype = np.dtype('>u2')  # Big-endian unsigned 16-bit integer
        shape = (2048, 2048)
        data = np.fromfile(fid, dtype)
        image = data.reshape(shape)

    # Apply thresholding and normalization
    threshold1 = 0
    threshold2 = 3000
    image[image < threshold1] = threshold1
    image[image > threshold2] = threshold2
    image = (image.astype(np.float32) - threshold1) / threshold2
    image = 1 - image  # Invert the image

    # Resize to 1024x1024
    image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_AREA)

    # Convert to 8-bit image
    image = (image * 255).astype(np.uint8)

    # Save as JPEG
    output_file = os.path.join(image_result_dir, file.replace('.IMG', '.jpg'))
    cv2.imwrite(output_file, image)
    print(f"Processed {file} -> {output_file}")

print("Finished processing images.")