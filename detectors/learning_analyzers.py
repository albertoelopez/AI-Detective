"""
Learning Analyzers - Educational signal processing techniques for AI detection

This module provides educational implementations of various analysis techniques
used to detect AI-generated content.
"""

import numpy as np
from PIL import Image
import cv2
import librosa
import pywt
import base64
from io import BytesIO
from typing import List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# WAVELET TRANSFORM
# ============================================================================

async def analyze_wavelet(file_path: str, media_type: str = "image") -> dict:
    """
    Perform wavelet transform analysis.

    Wavelets provide multi-resolution analysis - they show BOTH frequency content
    AND where in the signal those frequencies occur (unlike FFT which loses location).
    """
    if media_type == "image":
        return await _analyze_image_wavelet(file_path)
    elif media_type == "audio":
        return await _analyze_audio_wavelet(file_path)
    return {"error": "Unsupported media type"}


async def _analyze_image_wavelet(file_path: str) -> dict:
    """2D Wavelet decomposition of image."""
    img = Image.open(file_path).convert('L')
    img_array = np.array(img, dtype=np.float32)

    # Perform 2D wavelet decomposition (3 levels)
    coeffs = pywt.wavedec2(img_array, 'haar', level=3)

    # coeffs structure: [cA3, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1)]
    # cA = approximation (low freq), cH = horizontal detail, cV = vertical, cD = diagonal

    analysis = _analyze_wavelet_coeffs(coeffs)

    # Visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Original
    axes[0, 0].imshow(img_array, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Level 1 details
    axes[0, 1].imshow(np.abs(coeffs[3][0]), cmap='hot')
    axes[0, 1].set_title('Level 1: Horizontal Detail\n(vertical edges)')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(np.abs(coeffs[3][1]), cmap='hot')
    axes[0, 2].set_title('Level 1: Vertical Detail\n(horizontal edges)')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(np.abs(coeffs[3][2]), cmap='hot')
    axes[0, 3].set_title('Level 1: Diagonal Detail\n(diagonal edges)')
    axes[0, 3].axis('off')

    # Approximation and other levels
    axes[1, 0].imshow(coeffs[0], cmap='gray')
    axes[1, 0].set_title('Level 3: Approximation\n(low frequency)')
    axes[1, 0].axis('off')

    # Combined detail energy per level
    level_energies = []
    for i, (cH, cV, cD) in enumerate(coeffs[1:]):
        energy = np.sqrt(cH**2 + cV**2 + cD**2)
        level_energies.append(np.mean(energy))
        if i < 3:
            axes[1, i+1].imshow(energy, cmap='hot')
            axes[1, i+1].set_title(f'Level {3-i}: Combined Detail Energy')
            axes[1, i+1].axis('off')

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)

    return {
        "visualization": base64.b64encode(buf.read()).decode('utf-8'),
        "analysis": analysis,
        "explanation": {
            "title": "Understanding Wavelet Transform",
            "sections": [
                {
                    "heading": "What are Wavelets?",
                    "content": "Wavelets are like 'localized waves' that can analyze signals at multiple scales simultaneously. Unlike FFT which tells you WHAT frequencies exist, wavelets tell you WHAT frequencies exist WHERE."
                },
                {
                    "heading": "Decomposition Levels",
                    "content": "Each level splits the image into: Approximation (blurry version) and Details (horizontal, vertical, diagonal edges). Higher levels capture larger-scale features."
                },
                {
                    "heading": "Reading the Output",
                    "content": "Bright spots in detail images show edges. Level 1 captures fine details (noise, sharp edges). Level 3 captures large-scale structure. AI images often have unusual patterns in specific levels."
                },
                {
                    "heading": "AI Detection Clues",
                    "content": "1) Missing fine detail: Level 1 is unusually dark (over-smoothed). 2) Unusual level ratios: Energy distribution across levels differs from natural images. 3) Blocky artifacts: GAN artifacts appear as regular patterns in detail coefficients."
                }
            ]
        }
    }


def _analyze_wavelet_coeffs(coeffs) -> dict:
    """Analyze wavelet coefficients for AI indicators."""
    indicators = []
    ai_score = 0.0

    # Calculate energy at each level
    level_energies = []
    for level_coeffs in coeffs[1:]:  # Skip approximation
        cH, cV, cD = level_coeffs
        energy = np.mean(cH**2 + cV**2 + cD**2)
        level_energies.append(energy)

    # Normalize
    total_energy = sum(level_energies)
    if total_energy > 0:
        level_ratios = [e / total_energy for e in level_energies]
    else:
        level_ratios = [0.33, 0.33, 0.34]

    # Check 1: Fine detail deficiency (Level 1 should have significant energy in natural images)
    if level_ratios[2] < 0.15:  # Level 1 is index 2 (reversed order)
        indicators.append({
            "name": "Low fine detail energy",
            "description": "First level (fine details) has unusually low energy - common in AI over-smoothed images",
            "severity": "high"
        })
        ai_score += 0.35

    # Check 2: Unusual energy distribution
    expected_ratio = 0.33
    deviation = sum(abs(r - expected_ratio) for r in level_ratios) / 3
    if deviation > 0.15:
        indicators.append({
            "name": "Unusual scale distribution",
            "description": "Energy distribution across scales differs from natural images",
            "severity": "medium"
        })
        ai_score += 0.25

    # Check 3: Coefficient sparsity (natural images tend to have sparse wavelet coefficients)
    all_details = np.concatenate([np.abs(c).flatten() for level in coeffs[1:] for c in level])
    sparsity = np.sum(all_details < np.percentile(all_details, 90)) / len(all_details)

    if sparsity < 0.85:
        indicators.append({
            "name": "Dense wavelet coefficients",
            "description": "Natural images typically have sparser coefficients - dense patterns may indicate synthesis",
            "severity": "low"
        })
        ai_score += 0.15

    return {
        "level_energy_distribution": {
            f"level_{i+1}_percent": round(r * 100, 1) for i, r in enumerate(reversed(level_ratios))
        },
        "coefficient_sparsity": round(sparsity * 100, 1),
        "ai_indicators": indicators,
        "ai_likelihood_score": round(min(ai_score, 1.0) * 100, 1)
    }


async def _analyze_audio_wavelet(file_path: str) -> dict:
    """Wavelet analysis for audio."""
    y, sr = librosa.load(file_path, sr=22050, mono=True, duration=10)

    # Continuous wavelet transform
    scales = np.arange(1, 128)
    coefficients, frequencies = pywt.cwt(y[:sr*5], scales, 'morl', sampling_period=1/sr)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Waveform
    times = np.arange(len(y)) / sr
    axes[0, 0].plot(times[:sr*5], y[:sr*5], linewidth=0.5)
    axes[0, 0].set_title('Waveform')
    axes[0, 0].set_xlabel('Time (s)')

    # Scalogram (wavelet transform)
    im = axes[0, 1].imshow(np.abs(coefficients), aspect='auto', cmap='hot',
                           extent=[0, 5, frequencies[-1], frequencies[0]])
    axes[0, 1].set_title('Wavelet Scalogram')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Frequency (Hz)')
    plt.colorbar(im, ax=axes[0, 1])

    # Scale energy distribution
    scale_energy = np.mean(np.abs(coefficients)**2, axis=1)
    axes[1, 0].plot(frequencies, scale_energy)
    axes[1, 0].set_title('Energy by Frequency Scale')
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Energy')

    # Time-localized energy
    time_energy = np.mean(np.abs(coefficients)**2, axis=0)
    time_axis = np.linspace(0, 5, len(time_energy))
    axes[1, 1].plot(time_axis, time_energy)
    axes[1, 1].set_title('Energy Over Time')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Energy')

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)

    return {
        "visualization": base64.b64encode(buf.read()).decode('utf-8'),
        "analysis": {
            "duration_analyzed": 5.0,
            "scales_analyzed": len(scales),
            "ai_indicators": [],
            "ai_likelihood_score": 0
        },
        "explanation": {
            "title": "Audio Wavelet Analysis",
            "sections": [
                {
                    "heading": "Scalogram",
                    "content": "The scalogram shows how different frequencies appear over time. Unlike a spectrogram, wavelets have better time resolution at high frequencies and better frequency resolution at low frequencies."
                },
                {
                    "heading": "Why Wavelets for Audio?",
                    "content": "Transient sounds (clicks, consonants) are better captured by wavelets. AI audio sometimes has unnatural transients visible in wavelet analysis."
                }
            ]
        }
    }


# ============================================================================
# ERROR LEVEL ANALYSIS (ELA)
# ============================================================================

async def analyze_ela(file_path: str) -> dict:
    """
    Error Level Analysis - detects image manipulation through JPEG compression analysis.

    When a JPEG is resaved, the entire image should compress to similar error levels.
    Edited regions often have different error levels, revealing manipulation.
    """
    img = Image.open(file_path).convert('RGB')
    img_array = np.array(img)

    # Save at known quality and reload
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=95)
    buffer.seek(0)
    resaved = Image.open(buffer)
    resaved_array = np.array(resaved)

    # Calculate error levels
    ela = np.abs(img_array.astype(np.float32) - resaved_array.astype(np.float32))

    # Scale for visibility
    ela_scaled = (ela * 10).clip(0, 255).astype(np.uint8)

    analysis = _analyze_ela_result(ela)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_array)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(ela_scaled)
    axes[1].set_title('Error Level Analysis (ELA)\nBrighter = Higher Error Level')
    axes[1].axis('off')

    # Grayscale ELA with colormap
    ela_gray = np.mean(ela, axis=2)
    im = axes[2].imshow(ela_gray, cmap='hot')
    axes[2].set_title('ELA Heatmap\nInconsistent regions stand out')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)

    return {
        "visualization": base64.b64encode(buf.read()).decode('utf-8'),
        "analysis": analysis,
        "explanation": {
            "title": "Understanding Error Level Analysis",
            "sections": [
                {
                    "heading": "How ELA Works",
                    "content": "JPEG compression introduces predictable errors. When an image is resaved, the errors should be uniform. If parts were added/edited after the last save, those regions will have different error levels."
                },
                {
                    "heading": "Reading ELA Output",
                    "content": "Bright areas = high error levels. Dark areas = low error levels. In an unedited image, similar textures should have similar brightness. Inconsistencies suggest manipulation."
                },
                {
                    "heading": "AI Detection Application",
                    "content": "AI-generated images haven't been through real camera JPEG compression cycles. They often show unusual ELA patterns - either too uniform (synthetic) or with unexpected variations."
                },
                {
                    "heading": "Limitations",
                    "content": "ELA works best on JPEGs. Multiple resaves degrade the signal. High-contrast edges naturally show higher error levels. Use as one tool among many."
                }
            ]
        }
    }


def _analyze_ela_result(ela: np.ndarray) -> dict:
    """Analyze ELA result for manipulation indicators."""
    indicators = []
    ai_score = 0.0

    ela_gray = np.mean(ela, axis=2)

    # Check 1: Overall uniformity (AI images often too uniform)
    ela_std = np.std(ela_gray)
    ela_mean = np.mean(ela_gray)
    cv = ela_std / (ela_mean + 1e-6)

    if cv < 0.3:
        indicators.append({
            "name": "Unusually uniform error levels",
            "description": "ELA is very consistent across the image - may indicate synthetic generation",
            "severity": "medium"
        })
        ai_score += 0.3

    # Check 2: Look for distinct regions with different levels
    # Simple approach: check if there are clear bimodal distributions
    hist, bins = np.histogram(ela_gray.flatten(), bins=50)
    peaks = np.where((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:]))[0]

    if len(peaks) >= 2:
        indicators.append({
            "name": "Multiple distinct error levels",
            "description": "Image shows regions with clearly different compression histories",
            "severity": "high"
        })
        ai_score += 0.35

    # Check 3: Edge analysis - edges should have slightly higher ELA
    from scipy import ndimage
    edges = ndimage.sobel(np.mean(np.array(Image.open(BytesIO()).convert('L')), axis=None) if False else ela_gray)
    # Simplified: check correlation between edges and ELA

    return {
        "mean_error_level": round(ela_mean, 2),
        "error_level_variation": round(cv * 100, 1),
        "ai_indicators": indicators,
        "ai_likelihood_score": round(min(ai_score, 1.0) * 100, 1)
    }


# ============================================================================
# NOISE ANALYSIS
# ============================================================================

async def analyze_noise(file_path: str) -> dict:
    """
    Analyze image noise patterns.

    Real camera images have sensor noise with specific characteristics.
    AI-generated images often have synthetic or missing noise patterns.
    """
    img = Image.open(file_path).convert('RGB')
    img_array = np.array(img, dtype=np.float32)

    # Extract noise using various methods
    # Method 1: High-pass filter (subtracting blurred version)
    from scipy import ndimage
    blurred = ndimage.gaussian_filter(img_array, sigma=2)
    noise_hp = img_array - blurred

    # Method 2: Median filter residual
    median_filtered = ndimage.median_filter(img_array, size=3)
    noise_median = img_array - median_filtered

    analysis = _analyze_noise_patterns(noise_hp, noise_median, img_array)

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(img_array.astype(np.uint8))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Noise visualization (amplified)
    noise_vis = ((noise_hp - noise_hp.min()) / (noise_hp.max() - noise_hp.min() + 1e-6) * 255)
    axes[0, 1].imshow(noise_vis.astype(np.uint8))
    axes[0, 1].set_title('Extracted Noise (High-Pass)\nAmplified for visibility')
    axes[0, 1].axis('off')

    # Noise magnitude
    noise_mag = np.sqrt(np.sum(noise_hp**2, axis=2))
    im = axes[0, 2].imshow(noise_mag, cmap='hot')
    axes[0, 2].set_title('Noise Magnitude')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

    # Noise histogram
    axes[1, 0].hist(noise_hp.flatten(), bins=100, density=True, alpha=0.7)
    axes[1, 0].set_title('Noise Distribution\n(Should be roughly Gaussian)')
    axes[1, 0].set_xlabel('Noise Value')
    axes[1, 0].set_ylabel('Density')

    # Noise spectrum (FFT of noise)
    noise_gray = np.mean(noise_hp, axis=2)
    noise_fft = np.fft.fftshift(np.fft.fft2(noise_gray))
    noise_spectrum = np.log1p(np.abs(noise_fft))
    axes[1, 1].imshow(noise_spectrum, cmap='hot')
    axes[1, 1].set_title('Noise Spectrum (FFT)\nPatterns here indicate synthetic noise')
    axes[1, 1].axis('off')

    # Local noise variance map
    local_var = ndimage.generic_filter(noise_gray, np.var, size=16)
    axes[1, 2].imshow(local_var, cmap='viridis')
    axes[1, 2].set_title('Local Noise Variance\nShould be relatively uniform')
    axes[1, 2].axis('off')

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)

    return {
        "visualization": base64.b64encode(buf.read()).decode('utf-8'),
        "analysis": analysis,
        "explanation": {
            "title": "Understanding Noise Analysis",
            "sections": [
                {
                    "heading": "What is Image Noise?",
                    "content": "Every camera sensor produces random noise - slight variations in pixel values. This noise has characteristic patterns based on the sensor type, ISO setting, and lighting conditions."
                },
                {
                    "heading": "Extracting Noise",
                    "content": "We subtract a blurred version of the image to isolate the high-frequency noise component. Real sensor noise should be randomly distributed (Gaussian)."
                },
                {
                    "heading": "AI Detection Clues",
                    "content": "1) Missing noise: AI images often look 'too clean' - noise levels are unnaturally low. 2) Patterned noise: Regular patterns in noise suggest synthetic generation. 3) Inconsistent noise: Different regions having different noise characteristics indicates manipulation."
                },
                {
                    "heading": "Noise Spectrum",
                    "content": "Real sensor noise has a relatively flat spectrum (white noise). AI-generated noise may have peaks or patterns in the frequency domain."
                }
            ]
        }
    }


def _analyze_noise_patterns(noise_hp: np.ndarray, noise_median: np.ndarray, original: np.ndarray) -> dict:
    """Analyze noise for AI generation indicators."""
    indicators = []
    ai_score = 0.0

    noise_gray = np.mean(noise_hp, axis=2)

    # Check 1: Noise level (AI images often have very low noise)
    noise_std = np.std(noise_gray)
    if noise_std < 2.0:
        indicators.append({
            "name": "Very low noise level",
            "description": "Image has unusually little noise - may be AI-generated or heavily processed",
            "severity": "high"
        })
        ai_score += 0.35
    elif noise_std < 4.0:
        indicators.append({
            "name": "Low noise level",
            "description": "Noise level is lower than typical camera images",
            "severity": "medium"
        })
        ai_score += 0.2

    # Check 2: Noise distribution (should be approximately Gaussian)
    from scipy import stats
    _, p_value = stats.normaltest(noise_gray.flatten()[:10000])  # Sample for speed
    if p_value < 0.01:
        indicators.append({
            "name": "Non-Gaussian noise distribution",
            "description": "Noise doesn't follow expected sensor noise distribution",
            "severity": "medium"
        })
        ai_score += 0.25

    # Check 3: Noise consistency across image
    h, w = noise_gray.shape
    quadrants = [
        noise_gray[:h//2, :w//2],
        noise_gray[:h//2, w//2:],
        noise_gray[h//2:, :w//2],
        noise_gray[h//2:, w//2:]
    ]
    quad_stds = [np.std(q) for q in quadrants]
    std_variation = np.std(quad_stds) / (np.mean(quad_stds) + 1e-6)

    if std_variation > 0.3:
        indicators.append({
            "name": "Inconsistent noise across regions",
            "description": "Different parts of the image have different noise characteristics",
            "severity": "high"
        })
        ai_score += 0.3

    return {
        "noise_std": round(noise_std, 2),
        "noise_distribution_p_value": round(p_value, 4),
        "noise_consistency": round((1 - std_variation) * 100, 1),
        "ai_indicators": indicators,
        "ai_likelihood_score": round(min(ai_score, 1.0) * 100, 1)
    }


# ============================================================================
# HISTOGRAM ANALYSIS
# ============================================================================

async def analyze_histogram(file_path: str) -> dict:
    """
    Analyze color and brightness histograms.

    Natural images have smooth, continuous histograms.
    AI-generated or manipulated images often have gaps, spikes, or unnatural distributions.
    """
    img = Image.open(file_path).convert('RGB')
    img_array = np.array(img)

    # Convert to different color spaces
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    analysis = _analyze_histograms(img_array, hsv, gray)

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original image
    axes[0, 0].imshow(img_array)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # RGB histograms
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        axes[0, 1].hist(img_array[:,:,i].flatten(), bins=256, color=color, alpha=0.5, label=color.upper())
    axes[0, 1].set_title('RGB Histograms')
    axes[0, 1].set_xlabel('Pixel Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()

    # Grayscale histogram
    axes[0, 2].hist(gray.flatten(), bins=256, color='gray', alpha=0.7)
    axes[0, 2].set_title('Grayscale Histogram')
    axes[0, 2].set_xlabel('Pixel Value')

    # Hue histogram
    axes[1, 0].hist(hsv[:,:,0].flatten(), bins=180, color='purple', alpha=0.7)
    axes[1, 0].set_title('Hue Distribution')
    axes[1, 0].set_xlabel('Hue (0-180)')

    # Saturation histogram
    axes[1, 1].hist(hsv[:,:,1].flatten(), bins=256, color='orange', alpha=0.7)
    axes[1, 1].set_title('Saturation Distribution')
    axes[1, 1].set_xlabel('Saturation (0-255)')

    # Value/Brightness histogram
    axes[1, 2].hist(hsv[:,:,2].flatten(), bins=256, color='yellow', alpha=0.7)
    axes[1, 2].set_title('Value/Brightness Distribution')
    axes[1, 2].set_xlabel('Value (0-255)')

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)

    return {
        "visualization": base64.b64encode(buf.read()).decode('utf-8'),
        "analysis": analysis,
        "explanation": {
            "title": "Understanding Histogram Analysis",
            "sections": [
                {
                    "heading": "What Histograms Show",
                    "content": "Histograms show the distribution of pixel values. Each bar represents how many pixels have a particular brightness or color value."
                },
                {
                    "heading": "RGB vs HSV",
                    "content": "RGB shows red, green, blue channels. HSV (Hue, Saturation, Value) separates color (hue) from intensity (value), often revealing manipulation more clearly."
                },
                {
                    "heading": "Natural Image Characteristics",
                    "content": "Real photos typically have smooth, continuous histograms without sharp gaps or spikes. The distribution depends on scene content but shouldn't have unnatural discontinuities."
                },
                {
                    "heading": "AI Detection Clues",
                    "content": "1) Gaps in histogram: Missing values suggest artificial processing. 2) Sharp spikes: Unusual peaks may indicate color manipulation. 3) Clipping: Harsh cutoffs at 0 or 255. 4) Too smooth: Unnaturally perfect distributions."
                }
            ]
        }
    }


def _analyze_histograms(rgb: np.ndarray, hsv: np.ndarray, gray: np.ndarray) -> dict:
    """Analyze histograms for AI indicators."""
    indicators = []
    ai_score = 0.0

    # Check 1: Gaps in histogram
    gray_hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    zero_bins = np.sum(gray_hist == 0)

    if zero_bins > 50:
        indicators.append({
            "name": "Histogram gaps detected",
            "description": f"{zero_bins} empty bins in grayscale histogram - suggests processing or generation artifacts",
            "severity": "medium"
        })
        ai_score += 0.25

    # Check 2: Unusual peaks (spikes)
    mean_count = np.mean(gray_hist)
    spikes = np.sum(gray_hist > mean_count * 5)

    if spikes > 5:
        indicators.append({
            "name": "Unusual histogram spikes",
            "description": "Multiple sharp peaks in distribution - may indicate color quantization or AI generation",
            "severity": "medium"
        })
        ai_score += 0.2

    # Check 3: Clipping analysis
    dark_clip = np.sum(gray < 5) / gray.size
    bright_clip = np.sum(gray > 250) / gray.size

    if dark_clip > 0.1 or bright_clip > 0.1:
        indicators.append({
            "name": "Significant clipping",
            "description": "Large portions of image at extreme values - may be over-processed",
            "severity": "low"
        })
        ai_score += 0.15

    # Check 4: Saturation analysis
    sat = hsv[:,:,1]
    sat_mean = np.mean(sat)

    if sat_mean > 180:
        indicators.append({
            "name": "Unusually high saturation",
            "description": "Colors are more saturated than typical photographs",
            "severity": "low"
        })
        ai_score += 0.15

    return {
        "histogram_gaps": int(zero_bins),
        "histogram_spikes": int(spikes),
        "mean_saturation": round(float(sat_mean), 1),
        "dark_clipping_percent": round(dark_clip * 100, 2),
        "bright_clipping_percent": round(bright_clip * 100, 2),
        "ai_indicators": indicators,
        "ai_likelihood_score": round(min(ai_score, 1.0) * 100, 1)
    }


# ============================================================================
# GRADIENT / EDGE ANALYSIS
# ============================================================================

async def analyze_gradient(file_path: str) -> dict:
    """
    Analyze gradients and edges in the image.

    AI-generated images often have edges that are too smooth, too sharp,
    or have inconsistent gradient patterns compared to real photographs.
    """
    img = Image.open(file_path).convert('RGB')
    img_array = np.array(img)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Compute gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_dir = np.arctan2(grad_y, grad_x)

    # Edge detection
    edges_canny = cv2.Canny(gray.astype(np.uint8), 50, 150)

    # Laplacian (second derivative - detects edges and texture)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    analysis = _analyze_gradient_patterns(grad_mag, grad_dir, laplacian, edges_canny)

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(img_array)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(grad_mag, cmap='hot')
    axes[0, 1].set_title('Gradient Magnitude\n(Edge strength)')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(grad_dir, cmap='hsv')
    axes[0, 2].set_title('Gradient Direction\n(Edge orientation)')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(edges_canny, cmap='gray')
    axes[1, 0].set_title('Canny Edges')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(np.abs(laplacian), cmap='hot')
    axes[1, 1].set_title('Laplacian (2nd derivative)\nTexture and edge detail')
    axes[1, 1].axis('off')

    # Gradient magnitude histogram
    axes[1, 2].hist(grad_mag.flatten(), bins=100, log=True)
    axes[1, 2].set_title('Gradient Magnitude Distribution')
    axes[1, 2].set_xlabel('Gradient Magnitude')
    axes[1, 2].set_ylabel('Frequency (log)')

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)

    return {
        "visualization": base64.b64encode(buf.read()).decode('utf-8'),
        "analysis": analysis,
        "explanation": {
            "title": "Understanding Gradient & Edge Analysis",
            "sections": [
                {
                    "heading": "What are Gradients?",
                    "content": "Gradients measure how quickly pixel values change. Strong gradients indicate edges. The gradient has both magnitude (how strong) and direction (which way)."
                },
                {
                    "heading": "Edge Detection",
                    "content": "Canny edge detection finds significant edges. The Laplacian finds areas of rapid intensity change. Together they reveal the structure of the image."
                },
                {
                    "heading": "Natural vs AI Edges",
                    "content": "Real photos have natural edge falloff due to optics and focus. AI images often have edges that are too uniform, too sharp, or unnaturally smooth transitions."
                },
                {
                    "heading": "AI Detection Clues",
                    "content": "1) Missing weak edges: Over-smoothed images lack subtle gradients. 2) Too-uniform edge strength: Natural images have varied edge intensities. 3) Unusual gradient distribution: AI may have different gradient statistics."
                }
            ]
        }
    }


def _analyze_gradient_patterns(grad_mag: np.ndarray, grad_dir: np.ndarray,
                                laplacian: np.ndarray, edges: np.ndarray) -> dict:
    """Analyze gradient patterns for AI indicators."""
    indicators = []
    ai_score = 0.0

    # Check 1: Edge density
    edge_density = np.sum(edges > 0) / edges.size

    if edge_density < 0.02:
        indicators.append({
            "name": "Low edge density",
            "description": "Image has very few detected edges - may be over-smoothed (common in AI images)",
            "severity": "medium"
        })
        ai_score += 0.25
    elif edge_density > 0.3:
        indicators.append({
            "name": "Very high edge density",
            "description": "Unusually many edges detected - may indicate artificial sharpening",
            "severity": "low"
        })
        ai_score += 0.1

    # Check 2: Gradient magnitude distribution
    grad_mean = np.mean(grad_mag)
    grad_std = np.std(grad_mag)
    grad_cv = grad_std / (grad_mean + 1e-6)

    if grad_cv < 1.0:
        indicators.append({
            "name": "Uniform gradient distribution",
            "description": "Gradients are unusually consistent - natural images have more variation",
            "severity": "medium"
        })
        ai_score += 0.25

    # Check 3: Laplacian variance (texture measure)
    laplacian_var = np.var(laplacian)

    if laplacian_var < 100:
        indicators.append({
            "name": "Low texture variance",
            "description": "Image lacks fine texture detail - common in AI-generated images",
            "severity": "high"
        })
        ai_score += 0.3

    return {
        "edge_density_percent": round(edge_density * 100, 2),
        "gradient_mean": round(float(grad_mean), 2),
        "gradient_cv": round(float(grad_cv), 2),
        "laplacian_variance": round(float(laplacian_var), 2),
        "ai_indicators": indicators,
        "ai_likelihood_score": round(min(ai_score, 1.0) * 100, 1)
    }


# ============================================================================
# AUTOCORRELATION
# ============================================================================

async def analyze_autocorrelation(file_path: str, media_type: str = "image") -> dict:
    """
    Autocorrelation analysis - detects repetitive patterns.

    By correlating a signal with itself at different offsets, we can find
    hidden periodicities that might indicate copy-paste, looping, or AI artifacts.
    """
    if media_type == "image":
        return await _analyze_image_autocorr(file_path)
    elif media_type == "audio":
        return await _analyze_audio_autocorr(file_path)
    return {"error": "Unsupported media type"}


async def _analyze_image_autocorr(file_path: str) -> dict:
    """2D autocorrelation of image."""
    img = Image.open(file_path).convert('L')
    img_array = np.array(img, dtype=np.float32)

    # Normalize
    img_norm = img_array - np.mean(img_array)

    # 2D autocorrelation via FFT
    f = np.fft.fft2(img_norm)
    autocorr = np.fft.ifft2(f * np.conj(f)).real
    autocorr = np.fft.fftshift(autocorr)

    # Normalize
    autocorr = autocorr / autocorr.max()

    analysis = _analyze_autocorr_2d(autocorr)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_array, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(autocorr, cmap='hot')
    axes[1].set_title('2D Autocorrelation\nCenter = zero lag')
    axes[1].axis('off')

    # Radial profile
    h, w = autocorr.shape
    center = (h//2, w//2)
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)

    radial_profile = np.bincount(r.ravel(), autocorr.ravel()) / np.bincount(r.ravel())
    axes[2].plot(radial_profile[:min(200, len(radial_profile))])
    axes[2].set_title('Radial Autocorrelation Profile')
    axes[2].set_xlabel('Lag (pixels)')
    axes[2].set_ylabel('Correlation')
    axes[2].axhline(y=0, color='gray', linestyle='--')

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)

    return {
        "visualization": base64.b64encode(buf.read()).decode('utf-8'),
        "analysis": analysis,
        "explanation": {
            "title": "Understanding Autocorrelation",
            "sections": [
                {
                    "heading": "What is Autocorrelation?",
                    "content": "Autocorrelation measures how similar a signal is to a shifted version of itself. High correlation at a specific offset means there's a repeating pattern at that interval."
                },
                {
                    "heading": "Reading the Output",
                    "content": "The center is always maximum (zero shift = perfect correlation). Patterns radiating from center indicate repeated structures. Distinct peaks away from center suggest periodic artifacts."
                },
                {
                    "heading": "AI Detection Application",
                    "content": "AI generators, especially GANs, can create repeating patterns (tiling artifacts). Copy-paste manipulation also shows as correlation peaks. Natural images typically have smooth, rapidly decaying autocorrelation."
                }
            ]
        }
    }


def _analyze_autocorr_2d(autocorr: np.ndarray) -> dict:
    """Analyze 2D autocorrelation for periodic patterns."""
    indicators = []
    ai_score = 0.0

    h, w = autocorr.shape
    center = (h//2, w//2)

    # Exclude central peak region
    mask = np.ones_like(autocorr, dtype=bool)
    y, x = np.ogrid[:h, :w]
    mask &= ((x - center[1])**2 + (y - center[0])**2) > 100

    # Find secondary peaks
    masked_autocorr = autocorr.copy()
    masked_autocorr[~mask] = 0

    max_secondary = np.max(masked_autocorr)

    if max_secondary > 0.3:
        indicators.append({
            "name": "Strong secondary correlation peaks",
            "description": "Significant repeating patterns detected - may indicate copy-paste or tiling artifacts",
            "severity": "high"
        })
        ai_score += 0.4
    elif max_secondary > 0.15:
        indicators.append({
            "name": "Moderate secondary peaks",
            "description": "Some repeating patterns detected",
            "severity": "medium"
        })
        ai_score += 0.2

    return {
        "max_secondary_correlation": round(float(max_secondary), 3),
        "ai_indicators": indicators,
        "ai_likelihood_score": round(min(ai_score, 1.0) * 100, 1)
    }


async def _analyze_audio_autocorr(file_path: str) -> dict:
    """Autocorrelation for audio."""
    y, sr = librosa.load(file_path, sr=22050, mono=True, duration=10)

    # Compute autocorrelation
    autocorr = np.correlate(y, y, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Full autocorrelation
    time_lags = np.arange(len(autocorr)) / sr
    axes[0].plot(time_lags[:sr*2], autocorr[:sr*2])
    axes[0].set_title('Audio Autocorrelation')
    axes[0].set_xlabel('Lag (seconds)')
    axes[0].set_ylabel('Correlation')

    # Zoomed to find pitch
    axes[1].plot(time_lags[:sr//10], autocorr[:sr//10])
    axes[1].set_title('Autocorrelation (Zoomed)\nPeaks indicate pitch period')
    axes[1].set_xlabel('Lag (seconds)')
    axes[1].set_ylabel('Correlation')

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)

    return {
        "visualization": base64.b64encode(buf.read()).decode('utf-8'),
        "analysis": {"ai_indicators": [], "ai_likelihood_score": 0},
        "explanation": {
            "title": "Audio Autocorrelation",
            "sections": [
                {
                    "heading": "Finding Pitch",
                    "content": "The first major peak after zero indicates the fundamental period of the sound. This can be used to detect pitch and find looping patterns."
                },
                {
                    "heading": "Detecting Loops",
                    "content": "Exact audio loops will show as perfect correlation at the loop interval. Slight variations help distinguish natural repetition from copy-paste."
                }
            ]
        }
    }


# ============================================================================
# OPTICAL FLOW
# ============================================================================

async def analyze_optical_flow(file_path: str) -> dict:
    """
    Optical flow analysis for video.

    Tracks motion between frames. Deepfakes often have unnatural motion patterns,
    especially around manipulated regions like faces.
    """
    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        return {"error": "Could not open video file"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read frames
    frames = []
    for i in range(min(60, frame_count)):
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
    cap.release()

    if len(frames) < 2:
        return {"error": "Not enough frames"}

    # Compute optical flow for several frame pairs
    flows = []
    flow_mags = []

    for i in range(1, min(len(frames), 30)):
        flow = cv2.calcOpticalFlowFarneback(
            frames[i-1], frames[i], None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        flows.append(flow)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_mags.append(mag)

    analysis = _analyze_optical_flow_patterns(flows, flow_mags)

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Sample frame
    axes[0, 0].imshow(frames[len(frames)//2], cmap='gray')
    axes[0, 0].set_title('Sample Frame')
    axes[0, 0].axis('off')

    # Flow magnitude for one pair
    if flow_mags:
        axes[0, 1].imshow(flow_mags[len(flow_mags)//2], cmap='hot')
        axes[0, 1].set_title('Optical Flow Magnitude')
        axes[0, 1].axis('off')

        # Flow visualization (HSV color wheel)
        flow = flows[len(flows)//2]
        hsv_flow = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv_flow[..., 0] = ang * 180 / np.pi / 2
        hsv_flow[..., 1] = 255
        hsv_flow[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb_flow = cv2.cvtColor(hsv_flow, cv2.COLOR_HSV2RGB)
        axes[0, 2].imshow(rgb_flow)
        axes[0, 2].set_title('Flow Direction (Color = Direction)')
        axes[0, 2].axis('off')

    # Flow magnitude over time
    mean_flow = [np.mean(fm) for fm in flow_mags]
    axes[1, 0].plot(mean_flow)
    axes[1, 0].set_title('Mean Flow Magnitude Over Time')
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylabel('Mean Motion')

    # Flow variance over time
    var_flow = [np.var(fm) for fm in flow_mags]
    axes[1, 1].plot(var_flow)
    axes[1, 1].set_title('Flow Variance Over Time')
    axes[1, 1].set_xlabel('Frame')
    axes[1, 1].set_ylabel('Motion Variance')

    # Histogram of flow magnitudes
    all_mags = np.concatenate([fm.flatten() for fm in flow_mags])
    axes[1, 2].hist(all_mags, bins=50, log=True)
    axes[1, 2].set_title('Flow Magnitude Distribution')
    axes[1, 2].set_xlabel('Flow Magnitude')

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)

    return {
        "visualization": base64.b64encode(buf.read()).decode('utf-8'),
        "analysis": analysis,
        "explanation": {
            "title": "Understanding Optical Flow",
            "sections": [
                {
                    "heading": "What is Optical Flow?",
                    "content": "Optical flow estimates the motion of pixels between consecutive frames. Each pixel gets a motion vector showing where it 'moved' to in the next frame."
                },
                {
                    "heading": "Reading the Visualization",
                    "content": "Magnitude (brightness) shows how much motion. Color shows direction. Smooth, consistent flow indicates natural motion. Discontinuities suggest problems."
                },
                {
                    "heading": "Deepfake Detection",
                    "content": "Face-swapped videos often have inconsistent flow around the face boundary. The synthetic face may move differently than the surrounding real footage."
                },
                {
                    "heading": "AI Video Artifacts",
                    "content": "AI-generated video may have: 1) Too-smooth flow (interpolation artifacts), 2) Flickering (inconsistent frame generation), 3) Unnatural motion patterns in generated regions."
                }
            ]
        }
    }


def _analyze_optical_flow_patterns(flows: list, flow_mags: list) -> dict:
    """Analyze optical flow for deepfake indicators."""
    indicators = []
    ai_score = 0.0

    if not flow_mags:
        return {"ai_indicators": [], "ai_likelihood_score": 0}

    # Check 1: Flow consistency over time
    mean_flows = [np.mean(fm) for fm in flow_mags]
    flow_variation = np.std(mean_flows) / (np.mean(mean_flows) + 1e-6)

    if flow_variation > 1.5:
        indicators.append({
            "name": "Highly variable motion",
            "description": "Motion intensity varies dramatically between frames - may indicate synthesis artifacts",
            "severity": "medium"
        })
        ai_score += 0.25

    # Check 2: Spatial flow consistency
    spatial_vars = [np.var(fm) for fm in flow_mags]
    avg_spatial_var = np.mean(spatial_vars)

    if avg_spatial_var < 5:
        indicators.append({
            "name": "Unnaturally uniform motion",
            "description": "Motion is too consistent across the frame - may indicate interpolation",
            "severity": "medium"
        })
        ai_score += 0.25

    return {
        "frames_analyzed": len(flow_mags),
        "temporal_variation": round(float(flow_variation), 3),
        "avg_spatial_variance": round(float(avg_spatial_var), 2),
        "ai_indicators": indicators,
        "ai_likelihood_score": round(min(ai_score, 1.0) * 100, 1)
    }


# ============================================================================
# CEPSTRAL ANALYSIS (MFCCs)
# ============================================================================

async def analyze_cepstral(file_path: str) -> dict:
    """
    Cepstral analysis using MFCCs (Mel-Frequency Cepstral Coefficients).

    MFCCs capture the spectral envelope of speech/audio in a compact form.
    They're widely used in speaker recognition and can distinguish synthetic voices.
    """
    y, sr = librosa.load(file_path, sr=22050, mono=True, duration=30)

    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Delta MFCCs (first derivative - captures dynamics)
    mfcc_delta = librosa.feature.delta(mfccs)

    # Delta-delta (second derivative)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

    # Compute mel spectrogram for reference
    S_mel = librosa.feature.melspectrogram(y=y, sr=sr)
    S_mel_db = librosa.power_to_db(S_mel, ref=np.max)

    analysis = _analyze_mfcc_patterns(mfccs, mfcc_delta)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Mel spectrogram
    librosa.display.specshow(S_mel_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[0, 0])
    axes[0, 0].set_title('Mel Spectrogram')

    # MFCCs
    librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=axes[0, 1])
    axes[0, 1].set_title('MFCCs\n(Cepstral Coefficients)')
    axes[0, 1].set_ylabel('MFCC Coefficient')

    # MFCC Deltas
    librosa.display.specshow(mfcc_delta, sr=sr, x_axis='time', ax=axes[1, 0])
    axes[1, 0].set_title('MFCC Deltas (Dynamics)')
    axes[1, 0].set_ylabel('Delta Coefficient')

    # MFCC statistics
    mfcc_means = np.mean(mfccs, axis=1)
    mfcc_stds = np.std(mfccs, axis=1)
    x = np.arange(len(mfcc_means))
    axes[1, 1].bar(x, mfcc_means, yerr=mfcc_stds, capsize=3)
    axes[1, 1].set_title('MFCC Statistics (Mean ± Std)')
    axes[1, 1].set_xlabel('MFCC Coefficient')
    axes[1, 1].set_ylabel('Value')

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)

    return {
        "visualization": base64.b64encode(buf.read()).decode('utf-8'),
        "analysis": analysis,
        "explanation": {
            "title": "Understanding Cepstral Analysis",
            "sections": [
                {
                    "heading": "What is the Cepstrum?",
                    "content": "The cepstrum is the 'spectrum of a spectrum'. It separates the source (vocal cords) from the filter (vocal tract shape). This reveals voice characteristics independent of pitch."
                },
                {
                    "heading": "MFCCs Explained",
                    "content": "MFCCs apply the mel scale (matching human hearing) before computing the cepstrum. The result is a compact representation of the spectral envelope - perfect for voice analysis."
                },
                {
                    "heading": "Reading MFCC Plots",
                    "content": "Lower coefficients (1-5) capture overall spectral shape. Higher coefficients capture fine detail. Deltas show how these change over time - important for speech dynamics."
                },
                {
                    "heading": "Synthetic Voice Detection",
                    "content": "AI voices often have: 1) Unusual MFCC statistics (means/variances differ from natural speech), 2) Too-consistent deltas (lack of natural variation), 3) Missing or artificial formant transitions."
                }
            ]
        }
    }


def _analyze_mfcc_patterns(mfccs: np.ndarray, mfcc_delta: np.ndarray) -> dict:
    """Analyze MFCC patterns for synthetic voice indicators."""
    indicators = []
    ai_score = 0.0

    # Check 1: MFCC variance (synthetic voices often have unusual variance)
    mfcc_vars = np.var(mfccs, axis=1)
    low_var_coeffs = np.sum(mfcc_vars[1:] < 1.0)  # Exclude first (energy)

    if low_var_coeffs > 5:
        indicators.append({
            "name": "Low MFCC variance",
            "description": "Several coefficients show unusually low variation - may indicate synthetic generation",
            "severity": "medium"
        })
        ai_score += 0.25

    # Check 2: Delta variance (dynamics)
    delta_vars = np.var(mfcc_delta, axis=1)
    low_delta_var = np.sum(delta_vars < 0.5)

    if low_delta_var > 5:
        indicators.append({
            "name": "Static MFCC dynamics",
            "description": "Deltas show little variation - speech may be too 'stable' for natural voice",
            "severity": "high"
        })
        ai_score += 0.35

    # Check 3: Correlation between coefficients
    mfcc_corr = np.corrcoef(mfccs)
    high_corr_pairs = np.sum(np.abs(mfcc_corr) > 0.8) - 13  # Exclude diagonal

    if high_corr_pairs > 20:
        indicators.append({
            "name": "Unusual MFCC correlation",
            "description": "Coefficients are more correlated than expected in natural speech",
            "severity": "medium"
        })
        ai_score += 0.2

    return {
        "mfcc_variance_avg": round(float(np.mean(mfcc_vars)), 3),
        "delta_variance_avg": round(float(np.mean(delta_vars)), 3),
        "low_variance_coefficients": int(low_var_coeffs),
        "ai_indicators": indicators,
        "ai_likelihood_score": round(min(ai_score, 1.0) * 100, 1)
    }
