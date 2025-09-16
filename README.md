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
python vrs_simulator.py -i <input_image> -o <output_image> -p <policy>
```

### Parameters

- `-i, --input`: Path to the native resolution input image (required)
- `-o, --output`: Path for the output simulated image (required)
- `-p, --policy`: VRS policy to apply (required)

### Available Policies

1. **`2x2`** - Standard 2x2 centroid sampling
2. **`4x4`** - Standard 4x4 centroid sampling
3. **`4x4_blend`** - Center-weighted blend (4x4 blocks with 4 samples)
4. **`4x4_gradient`** - Gradient propagation using bilinear interpolation
5. **`4x4_aware`** - Content-aware luminance sampling

### Examples

Test all policies on a single image:
```bash
# 2x2 standard centroid
python vrs_simulator.py -i scene.png -o scene_2x2.png -p 2x2

# 4x4 standard centroid
python vrs_simulator.py -i scene.png -o scene_4x4.png -p 4x4

# 4x4 center-weighted blend
python vrs_simulator.py -i scene.png -o scene_4x4_blend.png -p 4x4_blend

# 4x4 gradient propagation
python vrs_simulator.py -i scene.png -o scene_4x4_gradient.png -p 4x4_gradient

# 4x4 content-aware luminance
python vrs_simulator.py -i scene.png -o scene_4x4_aware.png -p 4x4_aware
```

### Generate Test Image

A test image generator is included for quick testing:
```bash
python test_image_generator.py
```

This creates `test_input.png` with various patterns suitable for VRS testing.

## Output

The tool provides:

1. **Simulated Image**: Saved to the specified output path
2. **Quality Metrics**: Printed to console
   - MSE (Mean Squared Error) - Lower is better
   - PSNR (Peak Signal-to-Noise Ratio) in dB - Higher is better
   - SSIM (Structural Similarity Index) - Higher is better (1.0 is perfect)

Example output:
```
Running policy: 4x4_blend...
Calculating image quality metrics...
  - MSE: 154.21 (Lower is better)
  - PSNR: 26.25 dB (Higher is better)
  - SSIM: 0.9581 (Higher is better, 1.0 is perfect)
Success! Image saved to scene_4x4_blend.png
```

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