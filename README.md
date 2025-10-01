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

1. **`2x2`** - Standard 2x2 centroid sampling
2. **`4x4`** - Standard 4x4 centroid sampling
3. **`4x4_blend`** - Center-weighted blend (4x4 blocks with 4 samples)
4. **`4x4_gradient`** - Gradient propagation using bilinear interpolation
5. **`4x4_aware`** - Content-aware luminance sampling

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
python vrs_simulator.py -i scene.png -o scene_2x2.png -p 2x2
```

**With hardware VRS (full comparison):**
```bash
# Compare simulation with hardware VRS output
python vrs_simulator.py -i native_scene.png -o sim_4x4.png -p 4x4 -hw hw_vrs_scene.png

# Test if gradient policy matches hardware better
python vrs_simulator.py -i native_scene.png -o sim_gradient.png -p 4x4_gradient -hw hw_vrs_scene.png
```

**Batch testing multiple policies:**
```bash
# Test all policies against the same hardware VRS
python vrs_simulator.py -i native.png -o sim_2x2.png -p 2x2 -hw hardware.png
python vrs_simulator.py -i native.png -o sim_4x4.png -p 4x4 -hw hardware.png
python vrs_simulator.py -i native.png -o sim_blend.png -p 4x4_blend -hw hardware.png
python vrs_simulator.py -i native.png -o sim_gradient.png -p 4x4_gradient -hw hardware.png
python vrs_simulator.py -i native.png -o sim_aware.png -p 4x4_aware -hw hardware.png
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

### Policy 1: Standard Centroid (2x2, 4x4)
- **Description**: Baseline policy mimicking common hardware VRS
- **Sampling**: Single sample from block center
- **Propagation**: Broadcasts color to all pixels in block

### Policy 2: Center-Weighted Blend (4x4_blend)
- **Description**: Hybrid policy for improved 4x4 quality
- **Sampling**: Four samples from 2x2 sub-quadrant centers
- **Propagation**: Averages samples and broadcasts result

### Policy 3: Content-Aware Luminance (4x4_aware)
- **Description**: Advanced policy preserving bright features
- **Sampling**: Samples brightest pixel in each block
- **Propagation**: Broadcasts brightest pixel's color

### Policy 4: Gradient Propagation (4x4_gradient)
- **Description**: High-quality policy eliminating blockiness
- **Sampling**: Four samples from block corners
- **Propagation**: Bilinear interpolation creates smooth gradients

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
    return vrs_image
```

2. Register it in `vrs_simulator.py`:
   - Add to the `choices` list in argument parser
   - Add a conditional branch in `main()` to call your function

## License

[Your License Here]