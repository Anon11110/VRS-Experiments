import cv2
import numpy as np


def nearest_neighbor_filtering_centroid(native_image, shading_rate):
    """
    Policy: Nearest-Neighbor Filtering Centroid
    Simulates hardware VRS with nearest-neighbor filtering by sampling the center pixel.
    One shader invocation per block at the integer center coordinate.

    Args:
        native_image: Input image in BGR format
        shading_rate: 2 for 2x2 blocks, 4 for 4x4 blocks

    Returns:
        Tuple of (simulated VRS image, sample count)
    """
    height, width, channels = native_image.shape
    vrs_image = native_image.copy()
    sample_count = 0

    for y in range(0, height, shading_rate):
        for x in range(0, width, shading_rate):
            sample_count += 1

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
    Policy: Corner Cycling
    Pick exactly one corner per block using a tiled cycling pattern,
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
    Policy: Content-Adaptive Corner
    For each block, evaluate the color gradient (ddx/ddy) at each corner,
    choose the corner with the smallest gradient, then broadcast that color.
    """
    height, width, _ = native_image.shape
    vrs_image = native_image.copy()
    sample_count = 0

    img = native_image.astype(np.float32)

    def grad_mag_color(y, x):
        """
        Calculates the gradient magnitude from BGR color vectors using
        a Sobel-like operator.
        """
        # 3x3 neighborhood indices with clamping
        x0, x1, x2 = max(x - 1, 0), x, min(x + 1, width - 1)
        y0, y1, y2 = max(y - 1, 0), y, min(y + 1, height - 1)

        # Get the 3-channel color vectors for the 3x3 neighborhood
        c00, c01, c02 = img[y0, x0], img[y0, x1], img[y0, x2]
        c10,       c12 = img[y1, x0],            img[y1, x2]
        c20, c21, c22 = img[y2, x0], img[y2, x1], img[y2, x2]

        # Apply Sobel operator to the color vectors to get gradient vectors
        # gx_vec will be a vector like [gx_B, gx_G, gx_R]
        gx_vec = (-1*c00 + 1*c02 - 2*c10 + 2*c12 - 1*c20 + 1*c22)
        gy_vec = (-1*c00 - 2*c01 - 1*c02 + 1*c20 + 2*c21 + 1*c22)

        # Sum the squared components of both gradient vectors
        # This is |gx_vec|^2 + |gy_vec|^2
        return np.sum(gx_vec**2) + np.sum(gy_vec**2)

    for y in range(0, height, shading_rate):
        for x in range(0, width, shading_rate):
            sample_count += 1
            block_height = min(shading_rate, height - y)
            block_width = min(shading_rate, width - x)

            corners = [
                (x,                   y),
                (x + block_width - 1, y),
                (x,                   y + block_height - 1),
                (x + block_width - 1, y + block_height - 1)
            ]

            # Compute gradient score at each corner, choose the smallest
            scores = [grad_mag_color(cy, cx) for (cx, cy) in corners]
            best_idx = int(np.argmin(scores))
            sx, sy = corners[best_idx]

            # Propagate the color from the "most stable" corner
            sampled_color = native_image[sy, sx]
            vrs_image[y:y+block_height, x:x+block_width] = sampled_color

    return vrs_image, sample_count


def sample_bilinear(image, x, y):
    """
    Samples a color from an image at a floating-point coordinate using bilinear interpolation.

    Args:
        image: Input image in BGR format
        x: Floating-point x coordinate
        y: Floating-point y coordinate

    Returns:
        Bilinearly interpolated color at (x, y)
    """
    height, width, _ = image.shape
    x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)

    c00 = image[y0, x0].astype(np.float64)
    c10 = image[y0, x1].astype(np.float64)
    c01 = image[y1, x0].astype(np.float64)
    c11 = image[y1, x1].astype(np.float64)

    u, v = x - x0, y - y0
    top = (1 - u) * c00 + u * c10
    bottom = (1 - u) * c01 + u * c11
    return (1 - v) * top + v * bottom


def bilinear_filtering_centroid(native_image, shading_rate=4):
    """
    Policy: Bilinear Filtering Centroid
    Simulates VRS by invoking a single shader at the center of each block,
    using bilinear interpolation for sub-pixel accuracy.

    Args:
        native_image: Input image in BGR format
        shading_rate: Block size (default: 4 for 4x4)

    Returns:
        Tuple of (simulated VRS image, sample count)
    """
    height, width, channels = native_image.shape
    vrs_image = np.zeros_like(native_image)
    sample_count = 0

    for y in range(0, height, shading_rate):
        for x in range(0, width, shading_rate):
            sample_count += 1

            block_height = min(shading_rate, height - y)
            block_width = min(shading_rate, width - x)

            # Calculate the sub-pixel coordinate for the center of the block
            center_x = x + (block_width - 1) / 2.0
            center_y = y + (block_height - 1) / 2.0

            # Sample the color from that single point using bilinear interpolation
            sampled_color = sample_bilinear(native_image, center_x, center_y)

            # Propagate that single color to the entire block
            vrs_image[y:y+block_height, x:x+block_width] = sampled_color

    return vrs_image, sample_count


def gradient_centroid(native_image, shading_rate=4):
    """
    Policy: Dynamic Gradient Centroid VRS
    Samples at the nearest integer pixel to the gradient-weighted centroid within each block.

    Gradient is computed using forward-difference approximations of GPU implicit
    derivatives:
        ddx(f)(x,y) ≈ f(x+1,y) - f(x,y)
        ddy(f)(x,y) ≈ f(x,y+1) - f(x,y)

    Args:
        native_image: Input image in BGR format
        shading_rate: Block size (default: 4 for 4x4)

    Returns:
        Tuple of (simulated VRS image, sample count)
    """
    height, width, channels = native_image.shape
    vrs_image = native_image.copy()
    sample_count = 0

    # Calculate Gradient Map using ddx/ddy (forward differences)
    gray_image = cv2.cvtColor(native_image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # ddx: f(x+1,y) - f(x,y)
    ddx = np.zeros_like(gray_image, dtype=np.float32)
    ddx[:, :-1] = gray_image[:, 1:] - gray_image[:, :-1]
    # Last column has no +x neighbor; set derivative to 0 (similar to helper-lane/edge behavior)
    ddx[:, -1] = 0.0

    # ddy: f(x,y+1) - f(x,y)
    ddy = np.zeros_like(gray_image, dtype=np.float32)
    ddy[:-1, :] = gray_image[1:, :] - gray_image[:-1, :]
    # Last row has no +y neighbor; set derivative to 0
    ddy[-1, :] = 0.0

    magnitude_map = np.hypot(ddx, ddy)

    # Create coordinate grids for a block to aid in centroid calculation
    local_y, local_x = np.mgrid[0:shading_rate, 0:shading_rate]

    for y in range(0, height, shading_rate):
        for x in range(0, width, shading_rate):
            sample_count += 1

            block_height = min(shading_rate, height - y)
            block_width = min(shading_rate, width - x)
            mag_block = magnitude_map[y:y+block_height, x:x+block_width]

            # Calculate Centroid (gradient-magnitude-weighted)
            total_magnitude = np.sum(mag_block)
            if total_magnitude > 1e-6:  # Avoid division by zero
                offset_y = np.sum(local_y[:block_height, :block_width] * mag_block) / total_magnitude
                offset_x = np.sum(local_x[:block_height, :block_width] * mag_block) / total_magnitude
            else:  # If block is flat, sample the center
                offset_y = (block_height - 1) / 2.0
                offset_x = (block_width - 1) / 2.0

            # Round to nearest integer pixel (nearest-neighbor)
            sample_y = min(int(round(y + offset_y)), height - 1)
            sample_x = min(int(round(x + offset_x)), width - 1)

            # Sample and Replicate
            sampled_color = native_image[sample_y, sample_x]
            vrs_image[y:y+block_height, x:x+block_width] = sampled_color

    return vrs_image, sample_count


def minimum_gradient(native_image, shading_rate=4):
    """
    Policy: Minimum Gradient
    Selects the pixel with the minimum gradient magnitude within each block.

    Uses ddx and ddy to mimic GPU shader derivative functions for gradient calculation.

    Args:
        native_image: Input image in BGR format
        shading_rate: Block size (default: 4 for 4x4)

    Returns:
        Tuple of (simulated VRS image, sample count)
    """
    height, width, channels = native_image.shape
    vrs_image = native_image.copy()
    sample_count = 0

    gray_image = cv2.cvtColor(native_image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Calculate ddx and ddy
    ddx = np.zeros_like(gray_image)
    ddy = np.zeros_like(gray_image)

    # ddx: horizontal gradient (difference with right neighbor)
    ddx[:, :-1] = gray_image[:, 1:] - gray_image[:, :-1]
    ddx[:, -1] = 0  # Edge case: last column has no right neighbor

    # ddy: vertical gradient (difference with bottom neighbor)
    ddy[:-1, :] = gray_image[1:, :] - gray_image[:-1, :]
    ddy[-1, :] = 0  # Edge case: last row has no bottom neighbor

    # Calculate gradient magnitude: sqrt(ddx^2 + ddy^2)
    magnitude_map = np.sqrt(ddx**2 + ddy**2)

    for y in range(0, height, shading_rate):
        for x in range(0, width, shading_rate):
            sample_count += 1

            # Calculate block boundaries
            block_height = min(shading_rate, height - y)
            block_width = min(shading_rate, width - x)

            # Extract gradient magnitude for this block
            mag_block = magnitude_map[y:y+block_height, x:x+block_width]

            # Find the pixel with minimum gradient (most stable/flat color)
            min_loc = np.unravel_index(np.argmin(mag_block), mag_block.shape)

            # Sample from the minimum gradient location
            sample_y = y + min_loc[0]
            sample_x = x + min_loc[1]
            sampled_color = native_image[sample_y, sample_x]

            # Propagate to all pixels in the block
            vrs_image[y:y+block_height, x:x+block_width] = sampled_color

    return vrs_image, sample_count


def minimum_gradient(native_image, shading_rate=4):
    """
    Policy: Maximum Gradient
    Selects the pixel with the maximum gradient magnitude within each block.

    Args:
        native_image: Input image in BGR format
        shading_rate: Block size (default: 4 for 4x4)

    Returns:
        Tuple of (simulated VRS image, sample count)
    """
    height, width, channels = native_image.shape
    vrs_image = native_image.copy()
    sample_count = 0

    gray_image = cv2.cvtColor(native_image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Calculate ddx and ddy
    ddx = np.zeros_like(gray_image)
    ddy = np.zeros_like(gray_image)

    # ddx: horizontal gradient (difference with right neighbor)
    ddx[:, :-1] = gray_image[:, 1:] - gray_image[:, :-1]
    ddx[:, -1] = 0  # Edge case: last column has no right neighbor

    # ddy: vertical gradient (difference with bottom neighbor)
    ddy[:-1, :] = gray_image[1:, :] - gray_image[:-1, :]
    ddy[-1, :] = 0  # Edge case: last row has no bottom neighbor

    # Calculate gradient magnitude: sqrt(ddx^2 + ddy^2)
    magnitude_map = np.sqrt(ddx**2 + ddy**2)

    # Process image in blocks
    for y in range(0, height, shading_rate):
        for x in range(0, width, shading_rate):
            sample_count += 1  # One shader invocation per block

            # Calculate block boundaries
            block_height = min(shading_rate, height - y)
            block_width = min(shading_rate, width - x)

            # Extract gradient magnitude for this block
            mag_block = magnitude_map[y:y+block_height, x:x+block_width]

            # Find the pixel with maximum gradient (edge/detail pixel)
            max_loc = np.unravel_index(np.argmax(mag_block), mag_block.shape)

            # Sample from the maximum gradient location
            sample_y = y + max_loc[0]
            sample_x = x + max_loc[1]
            sampled_color = native_image[sample_y, sample_x]

            # Propagate to all pixels in the block
            vrs_image[y:y+block_height, x:x+block_width] = sampled_color

    return vrs_image, sample_count