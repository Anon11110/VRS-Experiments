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


def corner_cycling(native_image, shading_rate=4, phase=0):
    """
    Policy 2: Corner Cycling
    Pick exactly one corner per block using a tiled 2×2 cycling pattern,
    then broadcast the sampled color to the whole block.

    Corner indices: 0=TL, 1=TR, 2=BL, 3=BR
    Pattern (per block, before phase):
        (bx%2, by%2) = (0,0)->0, (1,0)->1, (0,1)->2, (1,1)->3
    'phase' rotates this pattern: corner = (pattern + phase) & 3

    Args:
        native_image: Input image in BGR format
        shading_rate: Block size (default: 4 for 4x4)
        phase: Integer 0..3 (optional). Use different values per frame to "cycle".

    Returns:
        Tuple of (simulated VRS image, sample count)
    """
    height, width, channels = native_image.shape
    vrs_image = native_image.copy()
    sample_count = 0

    # Corner offsets inside a block for generic shading_rate
    # TL=(0,0), TR=(w-1,0), BL=(0,h-1), BR=(w-1,h-1)
    for y in range(0, height, shading_rate):
        for x in range(0, width, shading_rate):
            sample_count += 1

            block_height = min(shading_rate, height - y)
            block_width = min(shading_rate, width - x)

            bx = x // shading_rate
            by = y // shading_rate

            # 2×2 cycling pattern across blocks, then rotate by 'phase'
            pattern = (bx & 1) | ((by & 1) << 1)  # {0,1,2,3}
            corner = (pattern + (phase & 3)) & 3

            if corner == 0:        # TL
                sx, sy = x, y
            elif corner == 1:      # TR
                sx, sy = x + block_width - 1, y
            elif corner == 2:      # BL
                sx, sy = x, y + block_height - 1
            else:                  # BR
                sx, sy = x + block_width - 1, y + block_height - 1

            sampled_color = native_image[sy, sx]
            vrs_image[y:y+block_height, x:x+block_width] = sampled_color

    return vrs_image, sample_count


def content_adaptive_corner(native_image, shading_rate=4):
    """
    Policy 3: Content-Adaptive Corner (quality-focused)
    For each block, evaluate a small gradient magnitude around each corner,
    choose the corner with the smallest gradient, then broadcast that color.

    Args:
        native_image: Input image in BGR format
        shading_rate: Block size (default: 4 for 4x4)

    Returns:
        Tuple of (simulated VRS image, sample count)
    """
    height, width, channels = native_image.shape
    vrs_image = native_image.copy()
    sample_count = 0

    # Convert to luma (Y) for gradient eval; keep float for math
    # BGR to luma (BT.601-ish): Y = 0.114B + 0.587G + 0.299R
    img = native_image.astype(np.float32)
    luma = 0.114 * img[..., 0] + 0.587 * img[..., 1] + 0.299 * img[..., 2]

    # Simple Sobel-like gradient at a pixel (clamped)
    def grad_mag(y, x):
        # 3x3 neighborhood indices with clamping
        x0 = max(x - 1, 0)
        x1 = x
        x2 = min(x + 1, width - 1)
        y0 = max(y - 1, 0)
        y1 = y
        y2 = min(y + 1, height - 1)

        # Sobel kernels
        # Gx
        gx = (-1*luma[y0, x0] + 1*luma[y0, x2]
              -2*luma[y1, x0] + 2*luma[y1, x2]
              -1*luma[y2, x0] + 1*luma[y2, x2])
        # Gy
        gy = (-1*luma[y0, x0] - 2*luma[y0, x1] - 1*luma[y0, x2]
              +1*luma[y2, x0] + 2*luma[y2, x1] + 1*luma[y2, x2])

        return gx*gx + gy*gy  # squared magnitude is enough for ordering

    for y in range(0, height, shading_rate):
        for x in range(0, width, shading_rate):
            sample_count += 1

            block_height = min(shading_rate, height - y)
            block_width = min(shading_rate, width - x)

            # Four corner coordinates (TL, TR, BL, BR)
            corners = [
                (x,                      y                     ),  # 0
                (x + block_width - 1,    y                     ),  # 1
                (x,                      y + block_height - 1  ),  # 2
                (x + block_width - 1,    y + block_height - 1  )   # 3
            ]

            # Compute gradient score at each corner, choose the smallest
            scores = [grad_mag(cy, cx) for (cx, cy) in corners]
            best_idx = int(np.argmin(scores))
            sx, sy = corners[best_idx]

            sampled_color = native_image[sy, sx]
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


