import cv2
import numpy as np


def average_color(native_image, shading_rate):
    """
    Policy: Average Color (Default for 2x2 and 4x4)
    Simulates hardware VRS with bilinear filtering by averaging all pixels in each block.
    This is the most accurate simulation of real GPU VRS behavior.

    Args:
        native_image: Input image in BGR format
        shading_rate: 2 for 2x2 blocks, 4 for 4x4 blocks

    Returns:
        Tuple of (simulated VRS image, sample count)
    """
    height, width, channels = native_image.shape
    vrs_image = native_image.copy()
    sample_count = 0

    # Process image in blocks
    for y in range(0, height, shading_rate):
        for x in range(0, width, shading_rate):
            sample_count += 1  # One shader invocation per block

            # Calculate block boundaries
            block_height = min(shading_rate, height - y)
            block_width = min(shading_rate, width - x)

            # Extract the block
            block = native_image[y:y+block_height, x:x+block_width]

            # Calculate average color of all pixels in the block
            avg_color = np.mean(block, axis=(0, 1)).astype(np.uint8)

            # Propagate the average color to all pixels in the block
            vrs_image[y:y+block_height, x:x+block_width] = avg_color

    return vrs_image, sample_count


def standard_centroid(native_image, shading_rate):
    """
    Policy: Standard Centroid (Nearest-Neighbor)
    Simulates hardware VRS with nearest-neighbor filtering by sampling the center pixel.
    Less accurate than average_color but useful for testing nearest-neighbor scenarios.

    Args:
        native_image: Input image in BGR format
        shading_rate: 2 for 2x2 blocks, 4 for 4x4 blocks

    Returns:
        Tuple of (simulated VRS image, sample count)
    """
    height, width, channels = native_image.shape
    vrs_image = native_image.copy()
    sample_count = 0

    # Process image in blocks
    for y in range(0, height, shading_rate):
        for x in range(0, width, shading_rate):
            sample_count += 1  # One shader invocation per block

            # Calculate block boundaries
            block_height = min(shading_rate, height - y)
            block_width = min(shading_rate, width - x)

            # Sample from the center of the block
            sample_y = min(y + block_height // 2, height - 1)
            sample_x = min(x + block_width // 2, width - 1)
            sampled_color = native_image[sample_y, sample_x]

            # Propagate to all pixels in the block
            vrs_image[y:y+block_height, x:x+block_width] = sampled_color

    return vrs_image, sample_count


def center_weighted_blend(native_image):
    """
    Policy 2: Center-Weighted Blend
    Hybrid policy that blends multiple samples from block's interior for 4x4 blocks.

    Args:
        native_image: Input image in BGR format

    Returns:
        Tuple of (simulated VRS image, sample count)
    """
    height, width, channels = native_image.shape
    vrs_image = native_image.copy()
    shading_rate = 4
    sample_count = 0

    # Process image in 4x4 blocks
    for y in range(0, height, shading_rate):
        for x in range(0, width, shading_rate):
            sample_count += 1  # One blended result per block

            # Calculate block boundaries
            block_height = min(shading_rate, height - y)
            block_width = min(shading_rate, width - x)

            # Sample from centers of four 2x2 sub-quadrants
            samples = []

            # Top-left quadrant center (x+1, y+1)
            if y + 1 < height and x + 1 < width:
                samples.append(native_image[y + 1, x + 1].astype(np.float64))

            # Top-right quadrant center (x+3, y+1)
            if y + 1 < height and x + 3 < width:
                samples.append(native_image[y + 1, x + 3].astype(np.float64))

            # Bottom-left quadrant center (x+1, y+3)
            if y + 3 < height and x + 1 < width:
                samples.append(native_image[y + 3, x + 1].astype(np.float64))

            # Bottom-right quadrant center (x+3, y+3)
            if y + 3 < height and x + 3 < width:
                samples.append(native_image[y + 3, x + 3].astype(np.float64))

            # Average the samples (or use fallback if not enough samples)
            if samples:
                blended_color = np.mean(samples, axis=0).astype(np.uint8)
            else:
                # Fallback to center sample if block is too small
                sample_y = min(y + block_height // 2, height - 1)
                sample_x = min(x + block_width // 2, width - 1)
                blended_color = native_image[sample_y, sample_x]

            # Propagate to all pixels in the block
            vrs_image[y:y+block_height, x:x+block_width] = blended_color

    return vrs_image, sample_count


def content_aware_luminance(native_image):
    """
    Policy 3: Content-Aware Luminance
    Advanced policy that samples the brightest point within each 4x4 block.

    Args:
        native_image: Input image in BGR format

    Returns:
        Tuple of (simulated VRS image, sample count)
    """
    height, width, channels = native_image.shape
    vrs_image = native_image.copy()
    shading_rate = 4
    sample_count = 0

    # Convert to grayscale for luminance calculation
    gray_image = cv2.cvtColor(native_image, cv2.COLOR_BGR2GRAY)

    # Process image in 4x4 blocks
    for y in range(0, height, shading_rate):
        for x in range(0, width, shading_rate):
            sample_count += 1  # One shader invocation per block

            # Calculate block boundaries
            block_height = min(shading_rate, height - y)
            block_width = min(shading_rate, width - x)

            # Find the brightest pixel in the block
            block_gray = gray_image[y:y+block_height, x:x+block_width]
            max_loc = np.unravel_index(np.argmax(block_gray), block_gray.shape)

            # Sample from the brightest location
            sample_y = y + max_loc[0]
            sample_x = x + max_loc[1]
            sampled_color = native_image[sample_y, sample_x]

            # Propagate to all pixels in the block
            vrs_image[y:y+block_height, x:x+block_width] = sampled_color

    return vrs_image, sample_count


def gradient_propagation(native_image):
    """
    Policy 4: Gradient Propagation
    High-quality policy using bilinear interpolation from corner samples.

    Args:
        native_image: Input image in BGR format

    Returns:
        Tuple of (simulated VRS image, sample count)
    """
    height, width, channels = native_image.shape
    vrs_image = np.zeros_like(native_image, dtype=np.float64)
    shading_rate = 4
    sample_count = 0

    # Process image in 4x4 blocks
    for y in range(0, height, shading_rate):
        for x in range(0, width, shading_rate):
            sample_count += 1  # One interpolation operation per block

            # Calculate block boundaries
            block_height = min(shading_rate, height - y)
            block_width = min(shading_rate, width - x)

            # Sample from four corners
            # Top-left corner (x, y)
            tl_color = native_image[y, x].astype(np.float64)

            # Top-right corner (x+3, y) or rightmost valid pixel
            tr_x = min(x + shading_rate - 1, width - 1)
            tr_color = native_image[y, tr_x].astype(np.float64)

            # Bottom-left corner (x, y+3) or bottommost valid pixel
            bl_y = min(y + shading_rate - 1, height - 1)
            bl_color = native_image[bl_y, x].astype(np.float64)

            # Bottom-right corner (x+3, y+3) or bottom-right valid pixel
            br_x = min(x + shading_rate - 1, width - 1)
            br_y = min(y + shading_rate - 1, height - 1)
            br_color = native_image[br_y, br_x].astype(np.float64)

            # Perform bilinear interpolation for each pixel in the block
            for dy in range(block_height):
                for dx in range(block_width):
                    # Calculate interpolation weights
                    # Normalize to [0, 1] within the block
                    u = dx / max(block_width - 1, 1)
                    v = dy / max(block_height - 1, 1)

                    # Bilinear interpolation
                    top_interp = (1 - u) * tl_color + u * tr_color
                    bottom_interp = (1 - u) * bl_color + u * br_color
                    final_color = (1 - v) * top_interp + v * bottom_interp

                    vrs_image[y + dy, x + dx] = final_color

    # Convert back to uint8
    vrs_image = np.clip(vrs_image, 0, 255).astype(np.uint8)

    return vrs_image, sample_count


def contrast_adaptive_shading(native_image, shading_rate=2, threshold=100):
    """
    Policy 5: Contrast-Adaptive Shading (CAS)
    Dynamically reduces pixel shading rate in screen-space regions of low visual complexity.
    Applies coarse shading only to blocks with low luminance variance, preserving full resolution
    in high-detail areas.

    Args:
        native_image: Input image in BGR format
        shading_rate: Coarse shading rate for low-variance blocks (default: 2)
        threshold: Luminance variance threshold (lower = more conservative, higher = more aggressive)

    Returns:
        Tuple of (simulated VRS image, sample count)
    """
    height, width, channels = native_image.shape
    vrs_image = native_image.copy()
    sample_count = 0

    # Convert to grayscale for luminance variance calculation
    gray_image = cv2.cvtColor(native_image, cv2.COLOR_BGR2GRAY)

    # Process image in blocks
    for y in range(0, height, shading_rate):
        for x in range(0, width, shading_rate):
            # Calculate block boundaries
            block_height = min(shading_rate, height - y)
            block_width = min(shading_rate, width - x)

            # Extract grayscale block and calculate luminance variance
            block_gray = gray_image[y:y+block_height, x:x+block_width]
            variance = np.var(block_gray)

            # Apply coarse shading for low-variance blocks, native for high-variance
            if variance < threshold:
                sample_count += 1  # Coarse shaded block is 1 shader invocation
                # Low variance: apply coarse shading (average color)
                block_native = native_image[y:y+block_height, x:x+block_width]
                avg_color = np.mean(block_native, axis=(0, 1)).astype(np.uint8)
                vrs_image[y:y+block_height, x:x+block_width] = avg_color
            else:
                # High variance: native resolution (one shader invocation per pixel)
                sample_count += block_height * block_width

    return vrs_image, sample_count


def stochastic_sampling(native_image, shading_rate=4, seed=None):
    """
    Policy 6: Stochastic Sampling
    Improves perceptual quality of coarse shading by transforming structured blockiness
    into unstructured high-frequency noise. Samples from a pseudo-random location within
    each block instead of a fixed centroid.

    Args:
        native_image: Input image in BGR format
        shading_rate: Block size for coarse shading (default: 4)
        seed: Random seed for reproducibility (optional)

    Returns:
        Tuple of (simulated VRS image, sample count)
    """
    height, width, channels = native_image.shape
    vrs_image = native_image.copy()
    sample_count = 0

    # Initialize random number generator
    rng = np.random.default_rng(seed)

    # Process image in blocks
    for y in range(0, height, shading_rate):
        for x in range(0, width, shading_rate):
            sample_count += 1  # One shader invocation per block

            # Calculate block boundaries
            block_height = min(shading_rate, height - y)
            block_width = min(shading_rate, width - x)

            # Generate pseudo-random offset within the block
            rand_dy = rng.integers(0, block_height)
            rand_dx = rng.integers(0, block_width)

            # Sample from the random location
            sample_y = y + rand_dy
            sample_x = x + rand_dx

            sampled_color = native_image[sample_y, sample_x]

            # Propagate to all pixels in the block
            vrs_image[y:y+block_height, x:x+block_width] = sampled_color

    return vrs_image, sample_count