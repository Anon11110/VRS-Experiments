import cv2
import numpy as np


def standard_centroid(native_image, shading_rate):
    """
    Policy 1: Standard Centroid
    Baseline policy that mimics common hardware VRS implementation.

    Args:
        native_image: Input image in BGR format
        shading_rate: 2 for 2x2 blocks, 4 for 4x4 blocks

    Returns:
        Simulated VRS image
    """
    height, width, channels = native_image.shape
    vrs_image = native_image.copy()

    # Process image in blocks
    for y in range(0, height, shading_rate):
        for x in range(0, width, shading_rate):
            # Calculate block boundaries
            block_height = min(shading_rate, height - y)
            block_width = min(shading_rate, width - x)

            # Sample from the center of the block
            sample_y = min(y + block_height // 2, height - 1)
            sample_x = min(x + block_width // 2, width - 1)
            sampled_color = native_image[sample_y, sample_x]

            # Propagate to all pixels in the block
            vrs_image[y:y+block_height, x:x+block_width] = sampled_color

    return vrs_image


def center_weighted_blend(native_image):
    """
    Policy 2: Center-Weighted Blend
    Hybrid policy that blends multiple samples from block's interior for 4x4 blocks.

    Args:
        native_image: Input image in BGR format

    Returns:
        Simulated VRS image
    """
    height, width, channels = native_image.shape
    vrs_image = native_image.copy()
    shading_rate = 4

    # Process image in 4x4 blocks
    for y in range(0, height, shading_rate):
        for x in range(0, width, shading_rate):
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

    return vrs_image


def content_aware_luminance(native_image):
    """
    Policy 3: Content-Aware Luminance
    Advanced policy that samples the brightest point within each 4x4 block.

    Args:
        native_image: Input image in BGR format

    Returns:
        Simulated VRS image
    """
    height, width, channels = native_image.shape
    vrs_image = native_image.copy()
    shading_rate = 4

    # Convert to grayscale for luminance calculation
    gray_image = cv2.cvtColor(native_image, cv2.COLOR_BGR2GRAY)

    # Process image in 4x4 blocks
    for y in range(0, height, shading_rate):
        for x in range(0, width, shading_rate):
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

    return vrs_image


def gradient_propagation(native_image):
    """
    Policy 4: Gradient Propagation
    High-quality policy using bilinear interpolation from corner samples.

    Args:
        native_image: Input image in BGR format

    Returns:
        Simulated VRS image
    """
    height, width, channels = native_image.shape
    vrs_image = np.zeros_like(native_image, dtype=np.float64)
    shading_rate = 4

    # Process image in 4x4 blocks
    for y in range(0, height, shading_rate):
        for x in range(0, width, shading_rate):
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

    return vrs_image