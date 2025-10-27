# VRS Simulator

A command-line tool for simulating and comparing Variable Rate Shading (VRS) policies on static images. This tool enables rapid offline evaluation of different VRS sampling and propagation strategies.

## Usage

### Basic Command

```bash
python vrs_simulator.py -i <native_image> -o <output_image> -p <policy> [-hw <hardware_vrs_image>] [-pf <pre_final_pass_image>]
```

### Parameters

- `-i, --input`: Path to the native resolution input image (required)
- `-o, --output`: Path for the output simulated image (required)
- `-p, --policy`: VRS policy to apply (required)
- `-hw, --hardware`: Path to hardware VRS image for comparison (optional)
- `-pf, --pre-final`: Path to pre-final pass image for delta-based VRS evaluation (optional)
- `--save-delta`: Path to save the delta/difference image between native and VRS result (optional)
- `--fp16`: Enable fp16 precision simulation for bilinear policies (matches GPU R16G16B16A16_FLOAT)

### Available Policies

All policies support both 2x2 and 4x4 block sizes:

1. **`2x2_centroid_nearest_neighbor`** / **`4x4_centroid_nearest_neighbor`** - Centroid sampling (nearest-neighbor)
2. **`2x2_center_bilinear`** / **`4x4_center_bilinear`** - Center-point sampling with bilinear interpolation
3. **`2x2_corner_cycle`** / **`4x4_corner_cycle`** - Corner cycling with tiled 2×2 pattern
4. **`2x2_corner_adaptive`** / **`4x4_corner_adaptive`** - Content-adaptive corner selection (quality-focused)
5. **`2x2_gradient_centroid`** / **`4x4_gradient_centroid`** - Dynamic gradient centroid sampling
6. **`2x2_minimum_gradient`** / **`4x4_minimum_gradient`** - Minimum gradient magnitude sampling (safest/most robust)
7. **`2x2_maximum_gradient`** / **`4x4_maximum_gradient`** - Maximum gradient magnitude sampling (edge-preserving but risky)

### Examples

#### Standard Mode: Full Image VRS

When you have hardware VRS output from your GPU:

1. Capture a native resolution image from your renderer
2. Capture the hardware VRS output for the same frame
3. Run the simulator to compare different policies against both the native and hardware images

```bash
# Compare simulated policy with hardware VRS
python vrs_simulator.py -i native.png -o sim_4x4.png -p 2x2_centroid_nearest_neighbor -hw hardware_vrs.png
```

This will show three comparisons:
- Simulated VRS vs Native (quality loss from simulation)
- Hardware VRS vs Native (quality loss from real GPU)
- Simulated VRS vs Hardware VRS (how well simulation matches hardware)

#### Delta Mode: VRS on Final Render Pass Only

When you have a multi-pass renderer with accumulative blending and want to evaluate VRS on only the final pass:

1. Capture the native resolution final image (all passes accumulated)
2. Capture the image before the final render pass (pre-final accumulation)
3. Optionally capture hardware VRS output for the final image
4. Run the simulator in delta mode

```bash
# Evaluate VRS on final pass only
python vrs_simulator.py -i native_final.png -pf pre_final_pass.png -o output.png -p 4x4_minimum_gradient

# With hardware comparison
python vrs_simulator.py -i native_final.png -pf pre_final_pass.png -hw hardware_vrs_final.png -o output.png -p 4x4_minimum_gradient
```

In delta mode, the simulator:
1. Computes the delta: `final_pass = native - pre_final`
2. Applies the VRS policy to the delta
3. Reconstructs the final image: `output = pre_final + vrs_delta`
4. If hardware image is provided, it also processes it the same way for fair comparison

This allows you to isolate and evaluate the quality impact of VRS on just the final render pass.

#### Delta Visualization

To visualize the differences between the native and VRS-processed images, use the `--save-delta` option:

```bash
# Save delta visualization
python vrs_simulator.py -i native.png -o output.png -p 4x4_minimum_gradient --save-delta delta.png
```

This generates two visualization files:
1. **`delta.png`** - Amplified difference image (5x amplification for better visibility)
2. **`delta_heatmap.png`** - Color-coded heatmap where:
   - Blue/green areas: Low difference (good VRS quality)
   - Yellow/orange areas: Moderate difference
   - Red areas: High difference (potential quality issues)

The tool also outputs delta statistics including maximum and mean differences to help quantify the impact.

## Policy Descriptions

### Nearest-Neighbor Filtering Centroid (2x2_centroid_nearest_neighbor, 4x4_centroid_nearest_neighbor)
- **Description**: Basic VRS with nearest-neighbor filtering
- **Sampling**: Single sample from integer pixel at block center

### Bilinear Filtering Centroid (2x2_center_bilinear, 4x4_center_bilinear)
- **Description**: Direct simulation of one shader invocation at block center with sub-pixel accuracy
- **Sampling**: Single sample at sub-pixel center coordinate using bilinear interpolation

### Corner Cycling (4x4_corner_cycle)
- **Description**: Picks one corner per block using a tiled 2×2 cycling pattern
- **Sampling**: Single sample from one of four corners (TL, TR, BL, BR) based on block position

### Content-Adaptive Corner (4x4_corner_adaptive)
- **Description**: Corner selection based on gradient analysis
- **Sampling**: Evaluates Sobel gradient at each corner, selects corner with smallest gradient

### Gradient Centroid (4x4_gradient_centroid)
- **Description**: Samples at gradient-weighted centroid of each block
- **Sampling**: Calculates gradient magnitude map, computes centroid weighted by gradient, rounds to nearest integer pixel (nearest-neighbor)

### Minimum Gradient (4x4_minimum_gradient)
- **Description**: Selects pixel with minimum gradient
- **Sampling**: Uses GPU-style ddx/ddy derivatives to calculate gradient magnitude, selects pixel with minimum gradient (flattest/most stable color)

### Maximum Gradient (4x4_maximum_gradient)
- **Description**: Selects pixel with maximum gradient
- **Sampling**: Uses GPU-style ddx/ddy derivatives to calculate gradient magnitude, selects pixel with maximum gradient (edge/detail pixel)
