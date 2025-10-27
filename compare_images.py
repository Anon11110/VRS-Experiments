import cv2
import argparse
import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity


def calculate_metrics(image1, image2):
    """Calculate image quality metrics between two images."""
    img1_float = image1.astype(np.float64)
    img2_float = image2.astype(np.float64)

    # Calculate MSE
    mse = mean_squared_error(img1_float, img2_float)

    # Calculate PSNR
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = peak_signal_noise_ratio(image1, image2, data_range=255)

    # Calculate SSIM
    img1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    ssim = structural_similarity(img1_rgb, img2_rgb, channel_axis=-1, data_range=255)

    return {"MSE": mse, "PSNR": psnr, "SSIM": ssim}


def main():
    parser = argparse.ArgumentParser(description="Compare two images using quality metrics")
    parser.add_argument("-i1", "--image1", required=True, type=str,
                        help="Path to the first image")
    parser.add_argument("-i2", "--image2", required=True, type=str,
                        help="Path to the second image")

    args = parser.parse_args()

    image1 = cv2.imread(args.image1)
    if image1 is None:
        print(f"Error: Could not load image from {args.image1}")
        return 1

    image2 = cv2.imread(args.image2)
    if image2 is None:
        print(f"Error: Could not load image from {args.image2}")
        return 1

    if image1.shape != image2.shape:
        print(f"Error: Image dimensions don't match")
        print(f"  Image 1: {image1.shape}")
        print(f"  Image 2: {image2.shape}")
        return 1

    # Calculate and display metrics
    print("\n" + "="*60)
    print("IMAGE COMPARISON METRICS")
    print("="*60)
    print(f"\nImage 1: {args.image1}")
    print(f"Image 2: {args.image2}")
    print(f"Resolution: {image1.shape[1]}x{image1.shape[0]}")
    print("-" * 60)

    metrics = calculate_metrics(image1, image2)

    print(f"\nMSE:  {metrics['MSE']:8.2f}  (Lower is better, 0 = identical)")
    if metrics['PSNR'] == float('inf'):
        print(f"PSNR:      ∞ dB  (Perfect match)")
    else:
        print(f"PSNR: {metrics['PSNR']:8.2f} dB  (Higher is better)")
    print(f"SSIM: {metrics['SSIM']:8.4f}  (Higher is better, 1.0 = perfect)")

    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    if metrics['SSIM'] > 0.99:
        print("Images are nearly identical")
    elif metrics['SSIM'] > 0.95:
        print("Images are very similar (excellent quality)")
    elif metrics['SSIM'] > 0.85:
        print("Images are quite similar (good quality)")
    elif metrics['SSIM'] > 0.70:
        print("Images have noticeable differences (fair quality)")
    else:
        print("Images are significantly different (poor quality)")

    print("="*60 + "\n")

    return 0


if __name__ == "__main__":
    exit(main())