# VRS Offline Policy Verifier (VOPV)

A command-line tool for simulating and comparing Variable Rate Shading (VRS) policies on static images. This tool enables rapid offline evaluation of different VRS sampling and propagation strategies without modifying a real-time renderer.

## Features

- **Multiple VRS Policies**: Implements several distinct VRS policies for comparison
- **Visual Output**: Generates simulated images for each policy
- **Quantitative Metrics**: Calculates MSE, PSNR, and SSIM for objective quality assessment
- **Modular Design**: Easy to extend with custom policies

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd vrs_experiment
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or if using conda:
```bash
conda install opencv numpy scikit-image
```

## Usage

### Basic Command

```bash
python vrs_simulator.py -i <native_image> -o <output_image> -p <policy> [-hw <hardware_vrs_image>]
```

### Parameters

- `-i, --input`: Path to the native resolution input image (required)
- `-o, --output`: Path for the output simulated image (required)
- `-p, --policy`: VRS policy to apply (required)
- `-hw, --hardware`: Path to hardware VRS image for comparison (optional)

### Available Policies

**Core VRS Policies (Single Sample per Block):**
1. **`2x2_centroid_nearest_neighbor`** - 2x2 centroid sampling (nearest-neighbor)
2. **`4x4_centroid_nearest_neighbor`** - 4x4 centroid sampling (nearest-neighbor)
3. **`2x2_center_bilinear`** - 2x2 center-point sampling with bilinear interpolation **[RECOMMENDED]**
4. **`4x4_center_bilinear`** - 4x4 center-point sampling with bilinear interpolation **[RECOMMENDED]**

**Advanced VRS Policies (Single Sample per Block):**
5. **`4x4_corner_cycle`** - Corner cycling with tiled 2×2 pattern
6. **`4x4_corner_adaptive`** - Content-adaptive corner selection (quality-focused)
7. **`4x4_gradient_centroid`** - Dynamic gradient centroid sampling
8. **`4x4_minimum_gradient`** - Minimum gradient magnitude sampling (safest/most robust)

### Typical Workflow

When you have hardware VRS output from your GPU:

1. Capture a native resolution image from your renderer
2. Capture the hardware VRS output for the same frame
3. Run the simulator to compare different policies against both the native and hardware images

```bash
# Compare simulated policy with hardware VRS
python vrs_simulator.py -i native.png -o sim_4x4.png -p 4x4 -hw hardware_vrs.png
```

This will show three comparisons:
- Simulated VRS vs Native (quality loss from simulation)
- Hardware VRS vs Native (quality loss from real GPU)
- Simulated VRS vs Hardware VRS (how well simulation matches hardware)

### Examples

**Without hardware VRS (basic usage):**
```bash
# Test a single policy against native resolution
python vrs_simulator.py -i scene.png -o scene_4x4.png -p 4x4_centroid_nearest_neighbor
```

**With hardware VRS (full comparison):**
```bash
# Compare simulation with hardware VRS output
python vrs_simulator.py -i native_scene.png -o sim_4x4.png -p 4x4_centroid_nearest_neighbor -hw hw_vrs_scene.png

# Test if center bilinear policy matches hardware better
python vrs_simulator.py -i native_scene.png -o sim_center_bilinear.png -p 4x4_center_bilinear -hw hw_vrs_scene.png
```

**Batch testing multiple policies:**
```bash
# Test all policies against the same hardware VRS
python vrs_simulator.py -i native.png -o sim_2x2_nn.png -p 2x2_centroid_nearest_neighbor -hw hardware.png
python vrs_simulator.py -i native.png -o sim_4x4_nn.png -p 4x4_centroid_nearest_neighbor -hw hardware.png
python vrs_simulator.py -i native.png -o sim_2x2_bilinear.png -p 2x2_center_bilinear -hw hardware.png
python vrs_simulator.py -i native.png -o sim_4x4_bilinear.png -p 4x4_center_bilinear -hw hardware.png
python vrs_simulator.py -i native.png -o sim_corner_cycle.png -p 4x4_corner_cycle -hw hardware.png
python vrs_simulator.py -i native.png -o sim_corner_adaptive.png -p 4x4_corner_adaptive -hw hardware.png
python vrs_simulator.py -i native.png -o sim_gradient_centroid.png -p 4x4_gradient_centroid -hw hardware.png
python vrs_simulator.py -i native.png -o sim_minimum_gradient.png -p 4x4_minimum_gradient -hw hardware.png
```

### Generate Test Image

A test image generator is included for quick testing:
```bash
python test_image_generator.py
```

This creates `test_input.png` with various patterns suitable for VRS testing.

## Output

### Without Hardware VRS Image

When running without the `-hw` parameter, you get:
- Simulated VRS image saved to output path
- Quality metrics comparing simulated vs native

Example:
```
Running policy: 2x2...

============================================================
QUALITY METRICS COMPARISON
============================================================

Simulated VRS (2x2) vs Native Resolution:
------------------------------------------------------------
  MSE:     282.85  (Lower is better)
  PSNR:     23.62 dB  (Higher is better)
  SSIM:   0.8959  (Higher is better, 1.0 is perfect)
============================================================

Success! Simulated VRS image saved to scene_2x2.png
```

### With Hardware VRS Image

When running with the `-hw` parameter, you get three-way comparison:
- Simulated VRS vs Native (quality loss from simulation)
- Hardware VRS vs Native (quality loss from real GPU)
- Simulated VRS vs Hardware VRS (how accurately the policy models hardware)

Example:
```
Running policy: 4x4...

============================================================
QUALITY METRICS COMPARISON
============================================================

Simulated VRS (4x4) vs Native Resolution:
------------------------------------------------------------
  MSE:      756.32  (Lower is better)
  PSNR:      19.34 dB  (Higher is better)
  SSIM:    0.7842  (Higher is better, 1.0 is perfect)

Hardware VRS vs Native Resolution:
------------------------------------------------------------
  MSE:      789.45  (Lower is better)
  PSNR:      19.15 dB  (Higher is better)
  SSIM:    0.7756  (Higher is better, 1.0 is perfect)

Simulated VRS (4x4) vs Hardware VRS:
------------------------------------------------------------
  MSE:       45.67  (Lower is better - shows similarity)
  PSNR:      31.53 dB  (Higher is better)
  SSIM:    0.9823  (Higher is better, 1.0 is perfect)

============================================================
SUMMARY
============================================================
Simulated policy matches hardware: EXCELLENT (SSIM > 0.95)
============================================================

Success! Simulated VRS image saved to sim_4x4.png
```

The summary helps you quickly understand if your simulated policy accurately models the hardware behavior.

## Policy Descriptions

### Nearest-Neighbor Filtering Centroid (2x2_centroid_nearest_neighbor, 4x4_centroid_nearest_neighbor)
- **Description**: Basic VRS with nearest-neighbor filtering
- **Sampling**: Single sample from integer pixel at block center
- **Propagation**: Broadcasts color to all pixels in block
- **Samples per Block**: 1 (at integer coordinate)
- **Use Case**: Simple VRS simulation; fastest but may produce blocky artifacts

### Bilinear Filtering Centroid (2x2_center_bilinear, 4x4_center_bilinear) **[RECOMMENDED]**
- **Description**: Direct simulation of one shader invocation at block center with sub-pixel accuracy
- **Sampling**: Single sample at sub-pixel center coordinate using bilinear interpolation
- **Propagation**: Broadcasts sampled color to entire block
- **Samples per Block**: 1 (at sub-pixel coordinate with bilinear interpolation)
- **Use Case**: Most realistic VRS simulation; represents true "one shader per block" behavior

### Corner Cycling (4x4_corner_cycle)
- **Description**: Picks one corner per block using a tiled 2×2 cycling pattern
- **Sampling**: Single sample from one of four corners (TL, TR, BL, BR) based on block position
- **Propagation**: Broadcasts corner color to entire block
- **Samples per Block**: 1
- **Use Case**: Reduces banding by distributing corner samples spatially; can cycle phase per frame

### Content-Adaptive Corner (4x4_corner_adaptive)
- **Description**: Quality-focused corner selection based on gradient analysis
- **Sampling**: Evaluates Sobel gradient at each corner, selects corner with smallest gradient
- **Propagation**: Broadcasts selected corner color to entire block
- **Samples per Block**: 1 (chosen intelligently)
- **Use Case**: Reduces edge leakage and block artifacts by selecting smoothest corner

### Gradient Centroid (4x4_gradient_centroid)
- **Description**: Advanced policy that samples at gradient-weighted centroid of each block
- **Sampling**: Calculates gradient magnitude map, computes centroid weighted by gradient, rounds to nearest integer pixel (nearest-neighbor)
- **Propagation**: Broadcasts sampled color to entire block
- **Samples per Block**: 1 (at nearest pixel to gradient-weighted centroid)
- **Use Case**: Intelligent sampling that adapts to local image structure; samples near edges/details within each block without sub-pixel interpolation

### Minimum Gradient (4x4_minimum_gradient)
- **Description**: Safest and most robust gradient-based VRS policy
- **Sampling**: Uses GPU-style ddx/ddy derivatives to calculate gradient magnitude, selects pixel with minimum gradient (flattest/most stable color)
- **Propagation**: Broadcasts sampled color to entire block
- **Samples per Block**: 1 (at minimum gradient location)
- **Use Case**: Best for maintaining visual quality by avoiding sampling from sharp edges or highlights; minimizes risk of smearing artifacts

## Limitations

- **No Performance Data**: Execution time doesn't reflect actual GPU performance
- **Texture Aliasing**: May show more aliasing than real hardware VRS with automatic mipmap selection
- **Static Images Only**: Designed for still image analysis, not video sequences

## Project Structure

```
vrs_experiment/
├── vrs_simulator.py       # Main script with CLI and orchestration
├── policies.py            # VRS policy implementations
├── test_image_generator.py # Utility to create test images
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Extending with Custom Policies

To add a new policy:

1. Add a new function to `policies.py`:
```python
def custom_policy(native_image, shading_rate=4):
    # Your implementation
    sample_count = 0
    # ... count shader invocations
    return vrs_image, sample_count
```

2. Register it in `vrs_simulator.py`:
   - Add to the `choices` list in argument parser
   - Add a conditional branch in `main()` to call your function and unpack the tuple

## License

[Your License Here]