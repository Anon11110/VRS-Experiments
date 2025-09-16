import cv2
import argparse
import numpy as np
from pathlib import Path
import policies
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity


def calculate_metrics(original_image, simulated_image):
    """Calculate image quality metrics between original and simulated images."""
    # Convert images to float for accurate metrics calculation
    orig_float = original_image.astype(np.float64)
    sim_float = simulated_image.astype(np.float64)

    # Calculate MSE
    mse = mean_squared_error(orig_float, sim_float)

    # Calculate PSNR (handle edge case where MSE is 0)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = peak_signal_noise_ratio(original_image, simulated_image, data_range=255)

    # Calculate SSIM
    # Convert BGR to RGB for SSIM calculation
    orig_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    sim_rgb = cv2.cvtColor(simulated_image, cv2.COLOR_BGR2RGB)
    ssim = structural_similarity(orig_rgb, sim_rgb, channel_axis=-1, data_range=255)

    return {"MSE": mse, "PSNR": psnr, "SSIM": ssim}


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="VRS Offline Policy Verifier (VOPV)")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to the native resolution input image")
    parser.add_argument("-o", "--output", required=True, type=str,
                        help="Path for the output simulated image")
    parser.add_argument("-p", "--policy", required=True, type=str,
                        choices=["2x2", "4x4", "4x4_blend", "4x4_gradient", "4x4_aware"],
                        help="VRS policy to apply")

    args = parser.parse_args()

    # Load the native resolution image
    native_image = cv2.imread(args.input)
    if native_image is None:
        print(f"Error: Could not load image from {args.input}")
        return 1

    print(f"Running policy: {args.policy}...")

    # Apply the selected VRS policy
    if args.policy == "2x2":
        vrs_image = policies.standard_centroid(native_image, shading_rate=2)
    elif args.policy == "4x4":
        vrs_image = policies.standard_centroid(native_image, shading_rate=4)
    elif args.policy == "4x4_blend":
        vrs_image = policies.center_weighted_blend(native_image)
    elif args.policy == "4x4_gradient":
        vrs_image = policies.gradient_propagation(native_image)
    elif args.policy == "4x4_aware":
        vrs_image = policies.content_aware_luminance(native_image)
    else:
        print(f"Error: Unknown policy {args.policy}")
        return 1

    # Calculate image quality metrics
    print("Calculating image quality metrics...")
    metrics = calculate_metrics(native_image, vrs_image)

    # Print metrics
    print(f"  - MSE: {metrics['MSE']:.2f} (Lower is better)")
    if metrics['PSNR'] == float('inf'):
        print(f"  - PSNR: ∞ dB (Perfect match)")
    else:
        print(f"  - PSNR: {metrics['PSNR']:.2f} dB (Higher is better)")
    print(f"  - SSIM: {metrics['SSIM']:.4f} (Higher is better, 1.0 is perfect)")

    # Save the output image
    success = cv2.imwrite(args.output, vrs_image)
    if success:
        print(f"Success! Image saved to {args.output}")
    else:
        print(f"Error: Could not save image to {args.output}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())