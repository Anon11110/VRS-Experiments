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
                        choices=["2x2_centroid", "4x4_centroid", "2x2_center_bilinear",
                                 "4x4_center_bilinear", "4x4_corner_cycle", "4x4_corner_adaptive",
                                 "4x4_gradient_centroid"],
                        help="VRS policy to apply")
    parser.add_argument("-hw", "--hardware", type=str,
                        help="Path to hardware VRS image for comparison (optional)")

    args = parser.parse_args()

    # Load the native resolution image
    native_image = cv2.imread(args.input)
    if native_image is None:
        print(f"Error: Could not load image from {args.input}")
        return 1

    # Load hardware VRS image if provided
    hardware_image = None
    if args.hardware:
        hardware_image = cv2.imread(args.hardware)
        if hardware_image is None:
            print(f"Error: Could not load hardware VRS image from {args.hardware}")
            return 1

        # Check dimensions match
        if hardware_image.shape != native_image.shape:
            print(f"Error: Hardware VRS image dimensions {hardware_image.shape} don't match native image {native_image.shape}")
            return 1

    print(f"Running policy: {args.policy}...")

    # Apply the selected VRS policy
    if args.policy == "2x2_centroid":
        vrs_image, sample_count = policies.nearest_neighbor_filtering_centroid(native_image, shading_rate=2)
    elif args.policy == "4x4_centroid":
        vrs_image, sample_count = policies.nearest_neighbor_filtering_centroid(native_image, shading_rate=4)
    elif args.policy == "2x2_center_bilinear":
        vrs_image, sample_count = policies.bilinear_filtering_centroid(native_image, shading_rate=2)
    elif args.policy == "4x4_center_bilinear":
        vrs_image, sample_count = policies.bilinear_filtering_centroid(native_image, shading_rate=4)
    elif args.policy == "4x4_corner_cycle":
        vrs_image, sample_count = policies.corner_cycling(native_image, shading_rate=4, phase=0)
    elif args.policy == "4x4_corner_adaptive":
        vrs_image, sample_count = policies.content_adaptive_corner(native_image, shading_rate=4)
    elif args.policy == "4x4_gradient_centroid":
        vrs_image, sample_count = policies.gradient_centroid(native_image, shading_rate=4)
    else:
        print(f"Error: Unknown policy {args.policy}")
        return 1

    # Calculate performance metrics
    height, width, channels = native_image.shape
    total_pixels = height * width
    native_sample_count = total_pixels
    shading_reduction = ((native_sample_count - sample_count) / native_sample_count) * 100

    # Calculate and display metrics
    print("\n" + "="*60)
    print("PERFORMANCE METRICS")
    print("="*60)
    print(f"Native Resolution Samples:    {native_sample_count:,}")
    print(f"VRS Policy Samples:           {sample_count:,}")
    print(f"Shading Reduction:            {shading_reduction:.2f}%")
    print(f"Speedup Factor:               {native_sample_count / sample_count:.2f}x")

    print("\n" + "="*60)
    print("QUALITY METRICS COMPARISON")
    print("="*60)

    # Simulated VRS vs Native (ground truth)
    print(f"\nSimulated VRS ({args.policy}) vs Native Resolution:")
    print("-" * 60)
    metrics_sim_vs_native = calculate_metrics(native_image, vrs_image)
    print(f"  MSE:  {metrics_sim_vs_native['MSE']:8.2f}  (Lower is better)")
    if metrics_sim_vs_native['PSNR'] == float('inf'):
        print(f"  PSNR:      ∞ dB  (Perfect match)")
    else:
        print(f"  PSNR: {metrics_sim_vs_native['PSNR']:8.2f} dB  (Higher is better)")
    print(f"  SSIM: {metrics_sim_vs_native['SSIM']:8.4f}  (Higher is better, 1.0 is perfect)")

    # If hardware VRS image is provided, compare it
    if hardware_image is not None:
        print(f"\nHardware VRS vs Native Resolution:")
        print("-" * 60)
        metrics_hw_vs_native = calculate_metrics(native_image, hardware_image)
        print(f"  MSE:  {metrics_hw_vs_native['MSE']:8.2f}  (Lower is better)")
        if metrics_hw_vs_native['PSNR'] == float('inf'):
            print(f"  PSNR:      ∞ dB  (Perfect match)")
        else:
            print(f"  PSNR: {metrics_hw_vs_native['PSNR']:8.2f} dB  (Higher is better)")
        print(f"  SSIM: {metrics_hw_vs_native['SSIM']:8.4f}  (Higher is better, 1.0 is perfect)")

        print(f"\nSimulated VRS ({args.policy}) vs Hardware VRS:")
        print("-" * 60)
        metrics_sim_vs_hw = calculate_metrics(hardware_image, vrs_image)
        print(f"  MSE:  {metrics_sim_vs_hw['MSE']:8.2f}  (Lower is better - shows similarity)")
        if metrics_sim_vs_hw['PSNR'] == float('inf'):
            print(f"  PSNR:      ∞ dB  (Perfect match)")
        else:
            print(f"  PSNR: {metrics_sim_vs_hw['PSNR']:8.2f} dB  (Higher is better)")
        print(f"  SSIM: {metrics_sim_vs_hw['SSIM']:8.4f}  (Higher is better, 1.0 is perfect)")

        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Simulated policy matches hardware: ", end="")
        if metrics_sim_vs_hw['SSIM'] > 0.95:
            print("EXCELLENT (SSIM > 0.95)")
        elif metrics_sim_vs_hw['SSIM'] > 0.85:
            print("GOOD (SSIM > 0.85)")
        elif metrics_sim_vs_hw['SSIM'] > 0.70:
            print("FAIR (SSIM > 0.70)")
        else:
            print("POOR (SSIM < 0.70)")

    print("="*60 + "\n")

    # Save the output image
    success = cv2.imwrite(args.output, vrs_image)
    if success:
        print(f"Success! Simulated VRS image saved to {args.output}")
    else:
        print(f"Error: Could not save image to {args.output}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())