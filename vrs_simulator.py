import cv2
import argparse
import numpy as np
from pathlib import Path
import policies
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity


def calculate_metrics(original_image, simulated_image):
    """Calculate image quality metrics between original and simulated images."""
    orig_float = original_image.astype(np.float64)
    sim_float = simulated_image.astype(np.float64)

    # Calculate MSE
    mse = mean_squared_error(orig_float, sim_float)

    # Calculate PSNR
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = peak_signal_noise_ratio(original_image, simulated_image, data_range=255)

    # Calculate SSIM
    orig_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    sim_rgb = cv2.cvtColor(simulated_image, cv2.COLOR_BGR2RGB)
    ssim = structural_similarity(orig_rgb, sim_rgb, channel_axis=-1, data_range=255)

    return {"MSE": mse, "PSNR": psnr, "SSIM": ssim}


def main():
    parser = argparse.ArgumentParser(description="VRS Offline Policy Verifier (VOPV)")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to the native resolution input image")
    parser.add_argument("-o", "--output", required=True, type=str,
                        help="Path for the output simulated image")
    parser.add_argument("-p", "--policy", required=True, type=str,
                        choices=["2x2_centroid_nearest_neighbor", "4x4_centroid_nearest_neighbor",
                                 "2x2_center_bilinear", "4x4_center_bilinear",
                                 "2x2_corner_cycle", "4x4_corner_cycle",
                                 "2x2_corner_adaptive", "4x4_corner_adaptive",
                                 "2x2_gradient_centroid", "4x4_gradient_centroid",
                                 "2x2_minimum_gradient", "4x4_minimum_gradient",
                                 "2x2_maximum_gradient", "4x4_maximum_gradient"],
                        help="VRS policy to apply")
    parser.add_argument("-hw", "--hardware", type=str,
                        help="Path to hardware VRS image for comparison (optional)")
    parser.add_argument("-pf", "--pre-final", type=str,
                        help="Path to pre-final pass image for delta-based VRS (optional)")
    parser.add_argument("--save-delta", type=str,
                        help="Path to save the delta/difference image between native and VRS result (optional)")

    args = parser.parse_args()

    native_image = cv2.imread(args.input)
    if native_image is None:
        print(f"Error: Could not load image from {args.input}")
        return 1

    # Delta mode: compute the final pass contribution
    if args.pre_final:
        pre_final_image = cv2.imread(args.pre_final)
        if pre_final_image is None:
            print(f"Error: Could not load pre-final pass image from {args.pre_final}")
            return 1
        if pre_final_image.shape != native_image.shape:
            print(f"Error: Pre-final pass image dimensions {pre_final_image.shape} don't match native image {native_image.shape}")
            return 1

        print(f"Delta mode: Computing final pass contribution (native - pre-final)...")
        delta = native_image.astype(np.float32) - pre_final_image.astype(np.float32)
        print(f"Delta range: [{np.min(delta):.2f}, {np.max(delta):.2f}], mean: {np.mean(delta):.2f}")

        # Normalize delta to [0, 255] for VRS processing
        delta_min, delta_max = np.min(delta), np.max(delta)
        if delta_max - delta_min > 1e-6:
            native_image = ((delta - delta_min) / (delta_max - delta_min) * 255.0).astype(np.uint8)
        else:
            native_image = np.zeros_like(native_image, dtype=np.uint8)
        print(f"Processing normalized delta with VRS policy...")
    else:
        pre_final_image = None
        delta_min, delta_max = None, None

    hardware_image = None
    hardware_image_original = None
    if args.hardware:
        hardware_image_original = cv2.imread(args.hardware)
        if hardware_image_original is None:
            print(f"Error: Could not load hardware VRS image from {args.hardware}")
            return 1

        if hardware_image_original.shape != native_image.shape:
            print(f"Error: Hardware VRS image dimensions {hardware_image_original.shape} don't match native image {native_image.shape}")
            return 1

        # If in delta mode, also extract the delta from hardware image
        if args.pre_final:
            print(f"Delta mode: Computing hardware VRS delta (hardware - pre-final)...")
            hw_delta = hardware_image_original.astype(np.float32) - pre_final_image.astype(np.float32)
            # Normalize using the same range as the native delta for fair comparison
            if delta_max - delta_min > 1e-6:
                hardware_image = ((hw_delta - delta_min) / (delta_max - delta_min) * 255.0).astype(np.uint8)
            else:
                hardware_image = np.zeros_like(hardware_image_original, dtype=np.uint8)
        else:
            hardware_image = hardware_image_original

    print(f"Running policy: {args.policy}...")

    if args.policy == "2x2_centroid_nearest_neighbor":
        vrs_image, sample_count = policies.nearest_neighbor_filtering_centroid(native_image, shading_rate=2)
    elif args.policy == "4x4_centroid_nearest_neighbor":
        vrs_image, sample_count = policies.nearest_neighbor_filtering_centroid(native_image, shading_rate=4)
    elif args.policy == "2x2_center_bilinear":
        vrs_image, sample_count = policies.bilinear_filtering_centroid(native_image, shading_rate=2)
    elif args.policy == "4x4_center_bilinear":
        vrs_image, sample_count = policies.bilinear_filtering_centroid(native_image, shading_rate=4)
    elif args.policy == "2x2_corner_cycle":
        vrs_image, sample_count = policies.corner_cycling(native_image, shading_rate=2, phase=0)
    elif args.policy == "4x4_corner_cycle":
        vrs_image, sample_count = policies.corner_cycling(native_image, shading_rate=4, phase=0)
    elif args.policy == "2x2_corner_adaptive":
        vrs_image, sample_count = policies.content_adaptive_corner(native_image, shading_rate=2)
    elif args.policy == "4x4_corner_adaptive":
        vrs_image, sample_count = policies.content_adaptive_corner(native_image, shading_rate=4)
    elif args.policy == "2x2_gradient_centroid":
        vrs_image, sample_count = policies.gradient_centroid(native_image, shading_rate=2)
    elif args.policy == "4x4_gradient_centroid":
        vrs_image, sample_count = policies.gradient_centroid(native_image, shading_rate=4)
    elif args.policy == "2x2_minimum_gradient":
        vrs_image, sample_count = policies.minimum_gradient(native_image, shading_rate=2)
    elif args.policy == "4x4_minimum_gradient":
        vrs_image, sample_count = policies.minimum_gradient(native_image, shading_rate=4)
    elif args.policy == "2x2_maximum_gradient":
        vrs_image, sample_count = policies.maximum_gradient(native_image, shading_rate=2)
    elif args.policy == "4x4_maximum_gradient":
        vrs_image, sample_count = policies.maximum_gradient(native_image, shading_rate=4)
    else:
        print(f"Error: Unknown policy {args.policy}")
        return 1

    # Calculate performance metrics
    height, width, channels = native_image.shape
    total_pixels = height * width
    native_sample_count = total_pixels
    shading_reduction = ((native_sample_count - sample_count) / native_sample_count) * 100

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
        print(f"  MSE:  {metrics_sim_vs_hw['MSE']:8.2f}  (Lower is better)")
        if metrics_sim_vs_hw['PSNR'] == float('inf'):
            print(f"  PSNR:      ∞ dB  (Perfect match)")
        else:
            print(f"  PSNR: {metrics_sim_vs_hw['PSNR']:8.2f} dB  (Higher is better)")
        print(f"  SSIM: {metrics_sim_vs_hw['SSIM']:8.4f}  (Higher is better, 1.0 is perfect)")

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

    # If delta mode, denormalize and reconstruct final images
    if args.pre_final:
        print(f"\nDenormalizing VRS delta and reconstructing final image...")
        vrs_delta = vrs_image.astype(np.float32)
        if delta_max - delta_min > 1e-6:
            vrs_delta = (vrs_delta / 255.0) * (delta_max - delta_min) + delta_min
        else:
            vrs_delta = np.zeros_like(vrs_delta)

        # Reconstruct: pre_final + vrs_delta
        vrs_image = pre_final_image.astype(np.float32) + vrs_delta
        vrs_image = np.clip(vrs_image, 0, 255).astype(np.uint8)

        if hardware_image is not None:
            hw_delta = hardware_image.astype(np.float32)
            if delta_max - delta_min > 1e-6:
                hw_delta = (hw_delta / 255.0) * (delta_max - delta_min) + delta_min
            else:
                hw_delta = np.zeros_like(hw_delta)
            hardware_image = pre_final_image.astype(np.float32) + hw_delta
            hardware_image = np.clip(hardware_image, 0, 255).astype(np.uint8)

    success = cv2.imwrite(args.output, vrs_image)
    if success:
        print(f"Success! Simulated VRS image saved to {args.output}")
    else:
        print(f"Error: Could not save image to {args.output}")
        return 1

    # Save delta image if requested
    if args.save_delta:
        if args.pre_final:
            original_native_image = cv2.imread(args.input)
        else:
            original_native_image = native_image

        diff = np.abs(original_native_image.astype(np.float32) - vrs_image.astype(np.float32))

        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        print(f"\nDelta Image Statistics:")
        print(f"  Max difference: {max_diff:.2f}")
        print(f"  Mean difference: {mean_diff:.2f}")

        # Option 1: Amplified visualization (scales differences for better visibility)
        # Multiply by a factor to make subtle differences more visible
        amplification_factor = 5.0
        diff_amplified = np.clip(diff * amplification_factor, 0, 255).astype(np.uint8)

        # Option 2: Heat map visualization
        # Convert to grayscale showing magnitude of differences
        diff_gray = np.mean(diff, axis=2)  # Average across color channels
        diff_normalized = (diff_gray / max_diff * 255).astype(np.uint8) if max_diff > 0 else np.zeros_like(diff_gray, dtype=np.uint8)

        # Apply a colormap for better visualization (red = high difference, blue = low)
        diff_heatmap = cv2.applyColorMap(diff_normalized, cv2.COLORMAP_JET)

        delta_success = cv2.imwrite(args.save_delta, diff_amplified)

        heatmap_path = args.save_delta.rsplit('.', 1)
        if len(heatmap_path) == 2:
            heatmap_path = f"{heatmap_path[0]}_heatmap.{heatmap_path[1]}"
        else:
            heatmap_path = f"{args.save_delta}_heatmap"
        heatmap_success = cv2.imwrite(heatmap_path, diff_heatmap)

        if delta_success:
            print(f"Delta image saved to {args.save_delta} (amplified {amplification_factor}x)")
            if heatmap_success:
                print(f"Delta heatmap saved to {heatmap_path}")
        else:
            print(f"Error: Could not save delta image to {args.save_delta}")

    return 0


if __name__ == "__main__":
    exit(main())