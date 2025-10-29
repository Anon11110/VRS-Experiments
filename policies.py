import cv2
import numpy as np


def nearest_neighbor_filtering_centroid(native_image, shading_rate, upsample_params=None, use_fp16=False):
    """
    Policy: Nearest-Neighbor Filtering Centroid
    Simulates hardware VRS with nearest-neighbor filtering by sampling the center pixel.
    One shader invocation per block at the integer center coordinate.

    In upsample mode, runs the upsampling shader once per block with the centroid coordinate.

    Args:
        native_image: Input image in BGR format (low-res texture in upsample mode)
        shading_rate: 2 for 2x2 blocks, 4 for 4x4 blocks
        upsample_params: Optional tuple of (low_res_image, high_res_shape) for upsample mode

    Returns:
        Tuple of (simulated VRS image, sample count)
    """
    if upsample_params:
        # UPSAMPLE MODE: Run shader once per high-res block
        low_res_image, high_res_shape = upsample_params
        height, width, channels = high_res_shape
        vrs_image = np.zeros((height, width, channels), dtype=np.float32)
    else:
        # STANDARD MODE: Sample from native image
        height, width, channels = native_image.shape
        vrs_image = native_image.copy()

    sample_count = 0

    for y in range(0, height, shading_rate):
        for x in range(0, width, shading_rate):
            sample_count += 1

            # Calculate block boundaries
            block_height = min(shading_rate, height - y)
            block_width = min(shading_rate, width - x)

            # Sample from the center of the block (integer coordinates)
            sample_y = min(y + block_height // 2, height - 1)
            sample_x = min(x + block_width // 2, width - 1)

            if upsample_params:
                # Get pixel-center UV for this high-res pixel
                u = (sample_x + 0.5) / width
                v = (sample_y + 0.5) / height
                # Run the upsample shader logic (bilinear sample from low-res)
                sampled_color = sample_lod0_bilinear_uv(low_res_image, u, v, use_fp16=use_fp16)
            else:
                sampled_color = native_image[sample_y, sample_x]

            # Propagate to all pixels in the block
            vrs_image[y:y+block_height, x:x+block_width] = sampled_color

    # Convert back to uint8 if in standard mode
    if not upsample_params and native_image.dtype == np.uint8:
        vrs_image = vrs_image.astype(np.uint8)

    return vrs_image, sample_count


def corner_cycling(native_image, shading_rate, phase=0, upsample_params=None, use_fp16=False):
    """
    Policy: Corner Cycling
    Pick exactly one corner per block using a tiled cycling pattern,
    then broadcast the sampled color to the whole block.

    Corner indices: 0=TL, 1=TR, 2=BL, 3=BR
    Pattern (per block, before phase):
        (bx%2, by%2) = (0,0)->0, (1,0)->1, (0,1)->2, (1,1)->3
    'phase' rotates this pattern: corner = (pattern + phase) & 3

    In upsample mode, runs the upsampling shader once per block with the selected corner coordinate.

    Args:
        native_image: Input image in BGR format (low-res texture in upsample mode)
        shading_rate: Block size (2 or 4)
        phase: Integer 0..3 (optional). Use different values per frame to "cycle".
        upsample_params: Optional tuple of (low_res_image, high_res_shape) for upsample mode

    Returns:
        Tuple of (simulated VRS image, sample count)
    """
    if upsample_params:
        # UPSAMPLE MODE: Run shader once per high-res block
        low_res_image, high_res_shape = upsample_params
        height, width, channels = high_res_shape
        vrs_image = np.zeros((height, width, channels), dtype=np.float32)
    else:
        # STANDARD MODE: Sample from native image
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

            if upsample_params:
                # Get pixel-center UV for this high-res pixel
                u = (sx + 0.5) / width
                v = (sy + 0.5) / height
                # Run the upsample shader logic (bilinear sample from low-res)
                sampled_color = sample_lod0_bilinear_uv(low_res_image, u, v, use_fp16=use_fp16)
            else:
                sampled_color = native_image[sy, sx]

            vrs_image[y:y+block_height, x:x+block_width] = sampled_color

    # Convert back to uint8 if in standard mode
    if not upsample_params and native_image.dtype == np.uint8:
        vrs_image = vrs_image.astype(np.uint8)

    return vrs_image, sample_count


def content_adaptive_corner(native_image, shading_rate, upsample_params=None):
    """
    Policy: Content-Adaptive Corner
    For each block, evaluate the color gradient (ddx/ddy) at each corner,
    choose the corner with the smallest gradient, then broadcast that color.

    In upsample mode, analyzes gradients in the low-res texture to choose sampling location.

    Args:
        native_image: Input image in BGR format (low-res texture in upsample mode)
        shading_rate: Block size (2 or 4)
        upsample_params: Optional tuple of (low_res_image, high_res_shape) for upsample mode

    Returns:
        Tuple of (simulated VRS image, sample count)
    """
    if upsample_params:
        # UPSAMPLE MODE: Analyze low-res texture, output to high-res
        low_res_image, high_res_shape = upsample_params
        height, width, channels = high_res_shape
        vrs_image = np.zeros((height, width, channels), dtype=np.float32)

        # Analyze gradients in the LOW-RES texture
        low_height, low_width, _ = low_res_image.shape
        img = low_res_image.astype(np.float32)
    else:
        # STANDARD MODE: Analyze and output at same resolution
        height, width, _ = native_image.shape
        vrs_image = native_image.copy()
        img = native_image.astype(np.float32)
        low_height, low_width = height, width

    def grad_mag_color(y, x, img_width, img_height):
        """
        Calculates the gradient magnitude from BGR color vectors using
        a Sobel-like operator.
        """
        # 3x3 neighborhood indices with clamping
        x0, x1, x2 = max(x - 1, 0), x, min(x + 1, img_width - 1)
        y0, y1, y2 = max(y - 1, 0), y, min(y + 1, img_height - 1)

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

    sample_count = 0
    for y in range(0, height, shading_rate):
        for x in range(0, width, shading_rate):
            sample_count += 1
            block_height = min(shading_rate, height - y)
            block_width = min(shading_rate, width - x)

            if upsample_params:
                # Map high-res block corners to low-res texture coordinates
                # Calculate which low-res pixels correspond to the corners
                scale_x = low_width / width
                scale_y = low_height / height

                # High-res corner positions
                high_corners = [
                    (x,                   y),
                    (x + block_width - 1, y),
                    (x,                   y + block_height - 1),
                    (x + block_width - 1, y + block_height - 1)
                ]

                # Map to low-res texture space for gradient evaluation
                low_corners = [
                    (int((hx + 0.5) * scale_x), int((hy + 0.5) * scale_y))
                    for hx, hy in high_corners
                ]

                # Evaluate gradients at low-res positions
                scores = [grad_mag_color(ly, lx, low_width, low_height) for lx, ly in low_corners]
                best_idx = int(np.argmin(scores))

                # Use the high-res corner position for UV calculation
                sx, sy = high_corners[best_idx]

                # Get pixel-center UV for this high-res pixel
                u = (sx + 0.5) / width
                v = (sy + 0.5) / height

                # Sample from low-res texture
                sampled_color = sample_lod0_bilinear_uv(low_res_image, u, v, use_fp16=False)
            else:
                # Standard mode: corners are in native image space
                corners = [
                    (x,                   y),
                    (x + block_width - 1, y),
                    (x,                   y + block_height - 1),
                    (x + block_width - 1, y + block_height - 1)
                ]

                # Compute gradient score at each corner, choose the smallest
                scores = [grad_mag_color(cy, cx, width, height) for (cx, cy) in corners]
                best_idx = int(np.argmin(scores))
                sx, sy = corners[best_idx]

                # Propagate the color from the "most stable" corner
                sampled_color = native_image[sy, sx]

            vrs_image[y:y+block_height, x:x+block_width] = sampled_color

    # Convert back to uint8 if in standard mode
    if not upsample_params and native_image.dtype == np.uint8:
        vrs_image = vrs_image.astype(np.uint8)

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


def sample_lod0_bilinear_uv(image, u, v, use_fp16=False):
    """
    LOD0 bilinear sampling with normalized UV coordinates.
    Matches textureLodOffset(..., 0.0, ivec2(...)) with LINEAR filtering and CLAMP_TO_EDGE.

    Args:
        image: Input image (HxWxC numpy array)
        u: Normalized U coordinate [0, 1]
        v: Normalized V coordinate [0, 1]
        use_fp16: If True, simulate fp16 precision path

    Returns:
        Bilinearly interpolated color
    """
    height, width = image.shape[:2]

    # Half-texel aligned LOD0 bilinear
    # Convert normalized UV to texel space with half-pixel offset
    sx = u * width - 0.5
    sy = v * height - 0.5

    # Floor to get base texel indices
    ix = int(np.floor(sx))
    iy = int(np.floor(sy))

    # Fractional parts for bilinear weights
    fx = sx - ix
    fy = sy - iy

    # CLAMP_TO_EDGE addressing mode
    x0 = np.clip(ix, 0, width - 1)
    x1 = np.clip(ix + 1, 0, width - 1)
    y0 = np.clip(iy, 0, height - 1)
    y1 = np.clip(iy + 1, 0, height - 1)

    if use_fp16:
        # --- FP16 (mediump) Simulation Path ---
        calc_dtype = np.float16
        fx_f = calc_dtype(fx)
        fy_f = calc_dtype(fy)
        one = calc_dtype(1.0)

        c00 = image[y0, x0].astype(calc_dtype)
        c10 = image[y0, x1].astype(calc_dtype)
        c01 = image[y1, x0].astype(calc_dtype)
        c11 = image[y1, x1].astype(calc_dtype)

        one_minus_fx = one - fx_f
        one_minus_fy = one - fy_f

        interp_y0 = c00 * one_minus_fx + c10 * fx_f
        interp_y1 = c01 * one_minus_fx + c11 * fx_f

        result = interp_y0 * one_minus_fy + interp_y1 * fy_f

        return result.astype(np.float32)

    else:
        # --- FP32 (highp) Simulation Path ---
        c00 = image[y0, x0].astype(np.float32)
        c10 = image[y0, x1].astype(np.float32)
        c01 = image[y1, x0].astype(np.float32)
        c11 = image[y1, x1].astype(np.float32)

        result = (c00 * (1.0 - fx) + c10 * fx) * (1.0 - fy) + \
                 (c01 * (1.0 - fx) + c11 * fx) * fy

        return result


def bilinear_filtering_centroid(native_image, shading_rate, use_fp16=False, upsample_params=None):
    """
    Policy: Bilinear Filtering Centroid
    Faithfully simulates GPU VRS with LOD0 bilinear filtering:
    - Representative sample at block center's pixel-center
    - LOD0 bilinear with CLAMP_TO_EDGE, normalized UV
    - Optional fp16 precision simulation (matches R16G16B16A16_FLOAT)

    In upsample mode, runs the upsampling shader once per block with the centroid coordinate.

    Args:
        native_image: Input image in BGR format (low-res texture in upsample mode)
        shading_rate: Block size (2 for 2x2, 4 for 4x4)
        use_fp16: If True, simulate fp16 texel storage and filtering
        upsample_params: Optional tuple of (low_res_image, high_res_shape) for upsample mode

    Returns:
        Tuple of (simulated VRS image, sample count)
    """
    if upsample_params:
        # UPSAMPLE MODE: Run shader once per high-res block
        low_res_image, high_res_shape = upsample_params
        height, width, channels = high_res_shape
        vrs_image = np.zeros((height, width, channels), dtype=np.float32)
        sample_image = low_res_image
    else:
        # STANDARD MODE: Sample from native image
        height, width, channels = native_image.shape
        vrs_image = np.zeros((height, width, channels), dtype=np.float32)
        sample_image = native_image

    sample_count = 0

    for y in range(0, height, shading_rate):
        for x in range(0, width, shading_rate):
            sample_count += 1

            block_height = min(shading_rate, height - y)
            block_width = min(shading_rate, width - x)

            # Block center's pixel-center coordinate
            # This matches GPU shader behavior: center of the representative pixel
            cx = x + (block_width - 1) / 2.0 + 0.5
            cy = y + (block_height - 1) / 2.0 + 0.5

            # Convert to normalized UV coordinates [0, 1]
            u = cx / width
            v = cy / height

            # LOD0 bilinear sample with CLAMP_TO_EDGE
            sampled_color = sample_lod0_bilinear_uv(sample_image, u, v, use_fp16)

            # Broadcast the sampled color to the entire block
            vrs_image[y:y+block_height, x:x+block_width] = sampled_color

    # Convert back to original dtype if needed (only in standard mode)
    if not upsample_params and native_image.dtype == np.uint8:
        vrs_image = np.clip(vrs_image, 0, 255).astype(np.uint8)
    elif not upsample_params and use_fp16:
        vrs_image = vrs_image.astype(np.float16).astype(native_image.dtype)

    return vrs_image, sample_count


def gradient_centroid(native_image, shading_rate, upsample_params=None):
    """
    Policy: Dynamic Gradient Centroid VRS
    Samples at the nearest integer pixel to the gradient-weighted centroid within each block.

    Gradient is computed using forward-difference approximations of GPU implicit
    derivatives:
        ddx(f)(x,y) ≈ f(x+1,y) - f(x,y)
        ddy(f)(x,y) ≈ f(x,y+1) - f(x,y)

    In upsample mode, analyzes gradients in the low-res texture to choose sampling location.

    Args:
        native_image: Input image in BGR format (low-res texture in upsample mode)
        shading_rate: Block size (default: 4 for 4x4)
        upsample_params: Optional tuple of (low_res_image, high_res_shape) for upsample mode

    Returns:
        Tuple of (simulated VRS image, sample count)
    """
    if upsample_params:
        # UPSAMPLE MODE: Analyze low-res texture, output to high-res
        low_res_image, high_res_shape = upsample_params
        height, width, channels = high_res_shape
        vrs_image = np.zeros((height, width, channels), dtype=np.float32)

        # Analyze gradients in the LOW-RES texture
        low_height, low_width, _ = low_res_image.shape
        gray_image = cv2.cvtColor(low_res_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        # STANDARD MODE: Analyze and output at same resolution
        height, width, channels = native_image.shape
        vrs_image = native_image.copy()
        gray_image = cv2.cvtColor(native_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        low_height, low_width = height, width

    sample_count = 0

    # Calculate Gradient Map using ddx/ddy (forward differences)
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

            if upsample_params:
                # Map high-res block to low-res texture coordinates
                scale_x = low_width / width
                scale_y = low_height / height

                # Calculate corresponding low-res block boundaries
                low_y_start = int(y * scale_y)
                low_x_start = int(x * scale_x)
                low_y_end = min(int((y + block_height) * scale_y), low_height)
                low_x_end = min(int((x + block_width) * scale_x), low_width)
                low_block_height = max(1, low_y_end - low_y_start)
                low_block_width = max(1, low_x_end - low_x_start)

                # Get gradient magnitudes for the corresponding low-res region
                mag_block = magnitude_map[low_y_start:low_y_end, low_x_start:low_x_end]

                # Calculate Centroid in low-res space (gradient-magnitude-weighted)
                total_magnitude = np.sum(mag_block)
                if total_magnitude > 1e-6 and mag_block.size > 0:  # Avoid division by zero
                    # Create local grids for the actual low-res block size
                    local_low_y, local_low_x = np.mgrid[0:low_block_height, 0:low_block_width]
                    offset_y = np.sum(local_low_y * mag_block) / total_magnitude
                    offset_x = np.sum(local_low_x * mag_block) / total_magnitude

                    # Convert low-res centroid to high-res coordinates
                    centroid_low_y = low_y_start + offset_y
                    centroid_low_x = low_x_start + offset_x

                    # Map back to high-res coordinates
                    sample_y = min(int(round(centroid_low_y / scale_y)), height - 1)
                    sample_x = min(int(round(centroid_low_x / scale_x)), width - 1)
                else:  # If block is flat or too small, sample the center
                    sample_y = min(y + block_height // 2, height - 1)
                    sample_x = min(x + block_width // 2, width - 1)

                # Get pixel-center UV for this high-res pixel
                u = (sample_x + 0.5) / width
                v = (sample_y + 0.5) / height

                # Sample from low-res texture
                sampled_color = sample_lod0_bilinear_uv(low_res_image, u, v, use_fp16=False)
            else:
                # Standard mode: work with native resolution gradient map
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

    # Convert back to uint8 if in standard mode
    if not upsample_params and native_image.dtype == np.uint8:
        vrs_image = vrs_image.astype(np.uint8)

    return vrs_image, sample_count


def minimum_gradient(native_image, shading_rate, upsample_params=None):
    """
    Policy: Minimum Gradient
    Selects the pixel with the minimum gradient magnitude within each block.

    Uses ddx and ddy to mimic GPU shader derivative functions for gradient calculation.

    In upsample mode, analyzes gradients in the low-res texture to choose sampling location.

    Args:
        native_image: Input image in BGR format (low-res texture in upsample mode)
        shading_rate: Block size (2 for 2x2, 4 for 4x4)
        upsample_params: Optional tuple of (low_res_image, high_res_shape) for upsample mode

    Returns:
        Tuple of (simulated VRS image, sample count)
    """
    if upsample_params:
        # UPSAMPLE MODE: Analyze low-res texture, output to high-res
        low_res_image, high_res_shape = upsample_params
        height, width, channels = high_res_shape
        vrs_image = np.zeros((height, width, channels), dtype=np.float32)

        # Analyze gradients in the LOW-RES texture
        low_height, low_width, _ = low_res_image.shape
        gray_image = cv2.cvtColor(low_res_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        # STANDARD MODE: Analyze and output at same resolution
        height, width, channels = native_image.shape
        vrs_image = native_image.copy()
        gray_image = cv2.cvtColor(native_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        low_height, low_width = height, width

    sample_count = 0

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

            if upsample_params:
                # Map high-res block to low-res texture coordinates
                scale_x = low_width / width
                scale_y = low_height / height

                # Calculate corresponding low-res block boundaries
                low_y_start = int(y * scale_y)
                low_x_start = int(x * scale_x)
                low_y_end = min(int((y + block_height) * scale_y), low_height)
                low_x_end = min(int((x + block_width) * scale_x), low_width)

                # Get gradient magnitudes for the corresponding low-res region
                if low_y_end > low_y_start and low_x_end > low_x_start:
                    mag_block = magnitude_map[low_y_start:low_y_end, low_x_start:low_x_end]

                    # Find the pixel with minimum gradient in low-res space
                    min_loc = np.unravel_index(np.argmin(mag_block), mag_block.shape)

                    # Convert low-res location to high-res coordinates
                    low_sample_y = low_y_start + min_loc[0]
                    low_sample_x = low_x_start + min_loc[1]

                    # Map back to high-res coordinates
                    sample_y = min(int(round(low_sample_y / scale_y)), height - 1)
                    sample_x = min(int(round(low_sample_x / scale_x)), width - 1)
                else:
                    # If low-res block is too small, sample the center
                    sample_y = min(y + block_height // 2, height - 1)
                    sample_x = min(x + block_width // 2, width - 1)

                # Get pixel-center UV for this high-res pixel
                u = (sample_x + 0.5) / width
                v = (sample_y + 0.5) / height

                # Sample from low-res texture
                sampled_color = sample_lod0_bilinear_uv(low_res_image, u, v, use_fp16=False)
            else:
                # Standard mode: work with native resolution gradient map
                mag_block = magnitude_map[y:y+block_height, x:x+block_width]

                # Find the pixel with minimum gradient (most stable/flat color)
                min_loc = np.unravel_index(np.argmin(mag_block), mag_block.shape)

                # Sample from the minimum gradient location
                sample_y = y + min_loc[0]
                sample_x = x + min_loc[1]
                sampled_color = native_image[sample_y, sample_x]

            # Propagate to all pixels in the block
            vrs_image[y:y+block_height, x:x+block_width] = sampled_color

    # Convert back to uint8 if in standard mode
    if not upsample_params and native_image.dtype == np.uint8:
        vrs_image = vrs_image.astype(np.uint8)

    return vrs_image, sample_count


def maximum_gradient(native_image, shading_rate, upsample_params=None):
    """
    Policy: Maximum Gradient
    Selects the pixel with the maximum gradient magnitude within each block.

    In upsample mode, analyzes gradients in the low-res texture to choose sampling location.

    Args:
        native_image: Input image in BGR format (low-res texture in upsample mode)
        shading_rate: Block size (2 for 2x2, 4 for 4x4)
        upsample_params: Optional tuple of (low_res_image, high_res_shape) for upsample mode

    Returns:
        Tuple of (simulated VRS image, sample count)
    """
    if upsample_params:
        # UPSAMPLE MODE: Analyze low-res texture, output to high-res
        low_res_image, high_res_shape = upsample_params
        height, width, channels = high_res_shape
        vrs_image = np.zeros((height, width, channels), dtype=np.float32)

        # Analyze gradients in the LOW-RES texture
        low_height, low_width, _ = low_res_image.shape
        gray_image = cv2.cvtColor(low_res_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        # STANDARD MODE: Analyze and output at same resolution
        height, width, channels = native_image.shape
        vrs_image = native_image.copy()
        gray_image = cv2.cvtColor(native_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        low_height, low_width = height, width

    sample_count = 0

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

            if upsample_params:
                # Map high-res block to low-res texture coordinates
                scale_x = low_width / width
                scale_y = low_height / height

                # Calculate corresponding low-res block boundaries
                low_y_start = int(y * scale_y)
                low_x_start = int(x * scale_x)
                low_y_end = min(int((y + block_height) * scale_y), low_height)
                low_x_end = min(int((x + block_width) * scale_x), low_width)

                # Get gradient magnitudes for the corresponding low-res region
                if low_y_end > low_y_start and low_x_end > low_x_start:
                    mag_block = magnitude_map[low_y_start:low_y_end, low_x_start:low_x_end]

                    # Find the pixel with maximum gradient in low-res space
                    max_loc = np.unravel_index(np.argmax(mag_block), mag_block.shape)

                    # Convert low-res location to high-res coordinates
                    low_sample_y = low_y_start + max_loc[0]
                    low_sample_x = low_x_start + max_loc[1]

                    # Map back to high-res coordinates
                    sample_y = min(int(round(low_sample_y / scale_y)), height - 1)
                    sample_x = min(int(round(low_sample_x / scale_x)), width - 1)
                else:
                    # If low-res block is too small, sample the center
                    sample_y = min(y + block_height // 2, height - 1)
                    sample_x = min(x + block_width // 2, width - 1)

                # Get pixel-center UV for this high-res pixel
                u = (sample_x + 0.5) / width
                v = (sample_y + 0.5) / height

                # Sample from low-res texture
                sampled_color = sample_lod0_bilinear_uv(low_res_image, u, v, use_fp16=False)
            else:
                # Standard mode: work with native resolution gradient map
                mag_block = magnitude_map[y:y+block_height, x:x+block_width]

                # Find the pixel with maximum gradient (edge/detail pixel)
                max_loc = np.unravel_index(np.argmax(mag_block), mag_block.shape)

                # Sample from the maximum gradient location
                sample_y = y + max_loc[0]
                sample_x = x + max_loc[1]
                sampled_color = native_image[sample_y, sample_x]

            # Propagate to all pixels in the block
            vrs_image[y:y+block_height, x:x+block_width] = sampled_color

    # Convert back to uint8 if in standard mode
    if not upsample_params and native_image.dtype == np.uint8:
        vrs_image = vrs_image.astype(np.uint8)

    return vrs_image, sample_count