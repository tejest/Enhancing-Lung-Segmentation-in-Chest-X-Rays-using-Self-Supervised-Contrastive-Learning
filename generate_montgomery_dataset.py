import numpy as np
import os
import shutil
import argparse
from PIL import Image
from skimage.io import imsave
import warnings

# Suppress low contrast warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Argument parsing
parser = argparse.ArgumentParser(description="Preprocess Indiana CXR dataset")
parser.add_argument("--data_dir", type=str, default=r'C:\Users\tejes\Downloads\indiana (1)\indiana\CXR_png')
parser.add_argument("--label_dir", type=str, default=r'C:\Users\tejes\Downloads\indiana (1)\indiana\GTMask')
parser.add_argument("--result_dir", type=str, default=r'C:\Users\tejes\Downloads\indiana (1)\indiana\converted_indiana')
args = parser.parse_args()

data_dir = args.data_dir
label_dir = args.label_dir
result_dir = args.result_dir

# Create result directories
os.makedirs(os.path.join(result_dir, 'image'), exist_ok=True)
os.makedirs(os.path.join(result_dir, 'label'), exist_ok=True)
image_result_dir = os.path.join(result_dir, 'image')
label_result_dir = os.path.join(result_dir, 'label')

n = 0
files = sorted(os.listdir(data_dir))

for file in files:
    if not file.endswith('.png'):
        print(f"Skipping {file}: Not a PNG file.")
        continue

    # Construct file paths
    image_path = os.path.join(data_dir, file)
    base_name = os.path.splitext(file)[0]
    left_mask_path = os.path.join(label_dir, 'leftMask', f"{base_name}.tif")
    right_mask_path = os.path.join(label_dir, 'rightMask', f"{base_name}.tif")
    single_mask_path = os.path.join(label_dir, 'single', f"{base_name}.tif")

    # Check if image and at least left and right masks exist
    if not (os.path.exists(image_path) and os.path.exists(left_mask_path) and os.path.exists(right_mask_path)):
        print(f"Skipping {file}: Missing image or required mask file(s).")
        continue

    # Check if single mask exists
    has_single = os.path.exists(single_mask_path)

    try:
        # Open image and masks
        image = Image.open(image_path).convert('RGB')
        image_left_lung = np.array(Image.open(left_mask_path).convert('L'))
        image_right_lung = np.array(Image.open(right_mask_path).convert('L'))
        image_single = np.array(Image.open(single_mask_path).convert('L')) if has_single else np.zeros_like(image_left_lung)

        # Combine masks (adjust threshold if necessary)
        mask = np.zeros_like(image_left_lung, dtype=np.uint8)
        mask[image_left_lung > 128] = 1  # Left lung
        mask[image_right_lung > 128] = 2  # Right lung
        if has_single:
            mask[image_single > 128] = 3  # Single mask

        # Save mask and copy image
        imsave(os.path.join(label_result_dir, f'label_{n:03d}.png'), mask)
        shutil.copy(image_path, os.path.join(image_result_dir, f'image_{n:03d}.png'))
        print(f"Processed {file} -> image_{n:03d}.png and label_{n:03d}.png")
        n += 1

    except Exception as e:
        print(f"Error processing {file}: {e}")

print(f"Finished processing {n} images.")
