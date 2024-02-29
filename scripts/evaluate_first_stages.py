import argparse
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread
from skimage.transform import resize
import subprocess


def compute_metrics(original_dir, reconstructed_dir, output_size=(256, 256)):
    psnr_values = []
    ssim_values = []

    for filename in os.listdir(original_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            continue  # Skip non-image files

        # Read the original and reconstructed images
        original_path = os.path.join(original_dir, filename)
        reconstructed_path = os.path.join(reconstructed_dir, filename)

        original_img = imread(original_path)
        reconstructed_img = imread(reconstructed_path)

        # Resize images to 256x256
        original_img = resize(original_img, output_size, anti_aliasing=True)
        reconstructed_img = resize(reconstructed_img, output_size, anti_aliasing=True)

        # Compute PSNR and SSIM
        psnr_value = psnr(original_img, reconstructed_img, data_range=original_img.max() - original_img.min())
        ssim_value = ssim(original_img, reconstructed_img, channel_axis=-1, data_range=original_img.max() - original_img.min())

        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)

    return np.mean(psnr_values), np.mean(ssim_values)


def calculate_rfid(image_dir1, image_dir2):
    fid_command = f'python -m pytorch_fid {image_dir1} {image_dir2}'
    fid_result = subprocess.run(fid_command, shell=True, capture_output=True, text=True)
    fid_score = float(fid_result.stdout.split(' ')[-1])
    return fid_score


def main(original_images_dir, reconstructed_images_dir1, reconstructed_images_dir2):
    resize_to = (256, 256)

    psnr1, ssim1 = compute_metrics(original_images_dir, reconstructed_images_dir1, output_size=resize_to)
    psnr2, ssim2 = compute_metrics(original_images_dir, reconstructed_images_dir2, output_size=resize_to)

    print(f"Model 1 - PSNR: {psnr1}, SSIM: {ssim1}")
    print(f"Model 2 - PSNR: {psnr2}, SSIM: {ssim2}")

    rfid1 = calculate_rfid(original_images_dir, reconstructed_images_dir1)
    rfid2 = calculate_rfid(original_images_dir, reconstructed_images_dir2)

    print(f"Model 1 - rFID: {rfid1}")
    print(f"Model 2 - rFID: {rfid2}")


if __name__ == "__main__":
    """
    python scripts/evaluate_first_stages.py \
        --original_dir ./eval_data \
        --reconstructed_dir1 ./reconstructed_images_pretrain \
        --reconstructed_dir2 /reconstructed_images_train200
    """
    parser = argparse.ArgumentParser(description="Evaluate models with PSNR, SSIM, and rFID")
    parser.add_argument('--original_dir', type=str, required=True, help='Directory of original images')
    parser.add_argument('--reconstructed_dir1', type=str, required=True,
                        help='Directory of reconstructed images from the first model')
    parser.add_argument('--reconstructed_dir2', type=str, required=True,
                        help='Directory of reconstructed images from the second model')

    args = parser.parse_args()
    main(args.original_dir, args.reconstructed_dir1, args.reconstructed_dir2)
