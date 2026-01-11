"""
AI-Generated Image Detector

Uses a combination of:
1. Hugging Face Inference API (umm-maybe/AI-image-detector)
2. Local feature analysis as fallback
"""
import httpx
import os
from PIL import Image
import numpy as np
from pathlib import Path

HF_API_URL = "https://api-inference.huggingface.co/models/umm-maybe/AI-image-detector"
HF_TOKEN = os.getenv("HF_TOKEN", "")


async def detect_ai_image(file_path: str) -> dict:
    """
    Detect if an image is AI-generated.

    Returns:
        dict with 'is_ai_generated', 'confidence', 'method', and 'details'
    """
    # Try Hugging Face API first (free tier: ~30k chars/month)
    try:
        result = await _detect_with_hf_api(file_path)
        if result:
            return result
    except Exception as e:
        print(f"HF API failed: {e}")

    # Fallback to local analysis
    return await _detect_with_local_analysis(file_path)


async def _detect_with_hf_api(file_path: str) -> dict | None:
    """Use Hugging Face Inference API for detection."""
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    with open(file_path, "rb") as f:
        data = f.read()

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(HF_API_URL, headers=headers, content=data)

        if response.status_code != 200:
            return None

        results = response.json()

        # Parse results - model returns [{"label": "artificial", "score": 0.99}, ...]
        ai_score = 0.0
        human_score = 0.0

        for item in results:
            if item["label"].lower() in ["artificial", "ai", "fake"]:
                ai_score = item["score"]
            elif item["label"].lower() in ["human", "real", "natural"]:
                human_score = item["score"]

        is_ai = ai_score > human_score
        confidence = ai_score if is_ai else human_score

        return {
            "is_ai_generated": is_ai,
            "confidence": round(confidence * 100, 2),
            "method": "huggingface_api",
            "model": "umm-maybe/AI-image-detector",
            "details": {
                "ai_score": round(ai_score * 100, 2),
                "human_score": round(human_score * 100, 2),
            }
        }


async def _detect_with_local_analysis(file_path: str) -> dict:
    """
    Local heuristic analysis for AI image detection.

    Looks for common AI artifacts:
    - Unusual color distribution
    - Repetitive patterns
    - Metadata anomalies
    - Frequency domain artifacts
    """
    img = Image.open(file_path)
    img_array = np.array(img.convert("RGB"))

    indicators = []
    score = 0.0

    # Check 1: Color histogram analysis
    # AI images often have unusual color distributions
    hist_score = _analyze_color_histogram(img_array)
    if hist_score > 0.6:
        indicators.append("unusual_color_distribution")
        score += 0.2

    # Check 2: High-frequency noise patterns
    noise_score = _analyze_noise_patterns(img_array)
    if noise_score > 0.5:
        indicators.append("synthetic_noise_patterns")
        score += 0.25

    # Check 3: Edge coherence (AI often has too-smooth or inconsistent edges)
    edge_score = _analyze_edges(img_array)
    if edge_score > 0.5:
        indicators.append("edge_anomalies")
        score += 0.2

    # Check 4: Metadata analysis
    metadata_score = _analyze_metadata(img)
    if metadata_score > 0.5:
        indicators.append("suspicious_metadata")
        score += 0.15

    # Check 5: Texture regularity
    texture_score = _analyze_texture(img_array)
    if texture_score > 0.5:
        indicators.append("artificial_texture_patterns")
        score += 0.2

    is_ai = score >= 0.4
    confidence = min(score * 100, 95)  # Cap at 95% for heuristic method

    return {
        "is_ai_generated": is_ai,
        "confidence": round(confidence, 2),
        "method": "local_heuristic",
        "details": {
            "indicators": indicators,
            "raw_score": round(score, 3),
            "note": "Heuristic analysis - less accurate than ML models"
        }
    }


def _analyze_color_histogram(img_array: np.ndarray) -> float:
    """Analyze color distribution for AI artifacts."""
    # AI images often have unusual peaks in color histograms
    scores = []
    for channel in range(3):
        hist, _ = np.histogram(img_array[:, :, channel], bins=256, range=(0, 256))
        hist = hist / hist.sum()

        # Check for unusual spikes (AI generators sometimes have quantization artifacts)
        peaks = np.sum(hist > 0.02)  # Count significant peaks
        scores.append(1.0 if peaks < 10 else 0.0)

        # Check for too-smooth distribution
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
        scores.append(1.0 if entropy < 6.0 else 0.0)

    return np.mean(scores)


def _analyze_noise_patterns(img_array: np.ndarray) -> float:
    """Detect synthetic noise patterns common in AI images."""
    # Convert to grayscale for noise analysis
    gray = np.mean(img_array, axis=2)

    # Calculate local variance
    from scipy import ndimage
    local_var = ndimage.generic_filter(gray, np.var, size=3)

    # AI images often have unnaturally uniform noise
    var_of_var = np.var(local_var)

    # Very low variance of variance suggests synthetic uniformity
    if var_of_var < 100:
        return 0.8
    elif var_of_var < 500:
        return 0.4
    return 0.1


def _analyze_edges(img_array: np.ndarray) -> float:
    """Analyze edge coherence for AI artifacts."""
    from scipy import ndimage

    gray = np.mean(img_array, axis=2)

    # Sobel edge detection
    sx = ndimage.sobel(gray, axis=0)
    sy = ndimage.sobel(gray, axis=1)
    edges = np.hypot(sx, sy)

    # AI images often have edges that are too uniform or have artifacts
    edge_std = np.std(edges)
    edge_mean = np.mean(edges)

    # Coefficient of variation
    cv = edge_std / (edge_mean + 1e-6)

    # Very low CV suggests overly uniform edges (AI artifact)
    if cv < 1.0:
        return 0.7
    elif cv < 1.5:
        return 0.4
    return 0.1


def _analyze_metadata(img: Image.Image) -> float:
    """Check metadata for AI generation indicators."""
    score = 0.0

    # Get EXIF data
    exif = img.getexif() if hasattr(img, 'getexif') else {}

    # No EXIF data is suspicious for "photos"
    if not exif:
        score += 0.3

    # Check for AI tool signatures in metadata
    info = img.info
    ai_indicators = ["stable diffusion", "midjourney", "dall-e", "comfyui",
                     "automatic1111", "novelai", "dream", "generated"]

    for key, value in info.items():
        if isinstance(value, str):
            value_lower = value.lower()
            if any(ind in value_lower for ind in ai_indicators):
                score += 0.7
                break

    return min(score, 1.0)


def _analyze_texture(img_array: np.ndarray) -> float:
    """Analyze texture for artificial patterns."""
    gray = np.mean(img_array, axis=2)

    # Calculate GLCM-like features (simplified)
    # Check for repetitive patterns
    h, w = gray.shape

    if h < 64 or w < 64:
        return 0.3

    # Sample patches and check similarity
    patches = []
    patch_size = 32
    for i in range(0, h - patch_size, patch_size):
        for j in range(0, w - patch_size, patch_size):
            patch = gray[i:i+patch_size, j:j+patch_size]
            patches.append(patch.flatten())

    if len(patches) < 4:
        return 0.3

    # Check correlation between patches
    patches = np.array(patches[:16])  # Limit to 16 patches
    correlations = np.corrcoef(patches)

    # High average correlation suggests repetitive AI patterns
    avg_corr = np.mean(np.abs(correlations[np.triu_indices(len(patches), k=1)]))

    if avg_corr > 0.8:
        return 0.8
    elif avg_corr > 0.6:
        return 0.5
    return 0.2
