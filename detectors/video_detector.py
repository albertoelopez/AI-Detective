"""
AI-Generated Video / Deepfake Detector

Detects deepfakes and AI-generated videos by analyzing:
1. Frame-level artifacts
2. Temporal consistency
3. Face manipulation indicators
4. Audio-visual sync (if applicable)
"""
import cv2
import numpy as np
from pathlib import Path
import tempfile
from typing import List, Tuple


async def detect_ai_video(file_path: str) -> dict:
    """
    Detect if a video is AI-generated or a deepfake.

    Returns:
        dict with 'is_ai_generated', 'confidence', 'method', and 'details'
    """
    try:
        cap = cv2.VideoCapture(file_path)

        if not cap.isOpened():
            return {
                "is_ai_generated": None,
                "confidence": 0,
                "method": "error",
                "error": "Could not open video file",
            }

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        # Sample frames for analysis (every Nth frame)
        sample_interval = max(1, frame_count // 30)  # Analyze ~30 frames max
        frames = []
        frame_indices = []

        for i in range(0, frame_count, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                frame_indices.append(i)
            if len(frames) >= 30:
                break

        cap.release()

        if len(frames) < 3:
            return {
                "is_ai_generated": None,
                "confidence": 0,
                "method": "error",
                "error": "Not enough frames to analyze",
            }

        # Run detection methods
        frame_result = _analyze_frame_artifacts(frames)
        temporal_result = _analyze_temporal_consistency(frames)
        face_result = _analyze_face_manipulation(frames)

        # Combine scores
        weights = {"frame": 0.30, "temporal": 0.35, "face": 0.35}

        combined_score = (
            frame_result["score"] * weights["frame"] +
            temporal_result["score"] * weights["temporal"] +
            face_result["score"] * weights["face"]
        )

        is_ai = combined_score > 0.45
        confidence = combined_score * 100

        indicators = []
        indicators.extend(frame_result.get("indicators", []))
        indicators.extend(temporal_result.get("indicators", []))
        indicators.extend(face_result.get("indicators", []))

        return {
            "is_ai_generated": is_ai,
            "confidence": round(confidence, 2),
            "method": "local_video_analysis",
            "details": {
                "frame_artifact_score": round(frame_result["score"] * 100, 2),
                "temporal_score": round(temporal_result["score"] * 100, 2),
                "face_manipulation_score": round(face_result["score"] * 100, 2),
                "indicators": indicators,
                "video_info": {
                    "duration_seconds": round(duration, 2),
                    "fps": round(fps, 2),
                    "total_frames": frame_count,
                    "frames_analyzed": len(frames),
                }
            }
        }

    except Exception as e:
        return {
            "is_ai_generated": None,
            "confidence": 0,
            "method": "error",
            "error": str(e),
        }


def _analyze_frame_artifacts(frames: List[np.ndarray]) -> dict:
    """
    Analyze individual frames for AI generation artifacts.

    Looks for:
    - Compression inconsistencies
    - Unnatural color patterns
    - Edge artifacts
    """
    indicators = []
    scores = []

    for frame in frames:
        frame_score = 0.0

        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Check for JPEG-like blocking artifacts inconsistencies
        # AI-generated content often has different compression patterns
        dct_score = _analyze_dct_artifacts(gray)
        if dct_score > 0.6:
            frame_score += 0.3

        # 2. Analyze color distribution
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_score = _analyze_color_artifacts(hsv)
        if color_score > 0.5:
            frame_score += 0.25

        # 3. Edge analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # Unusual edge patterns
        if edge_density < 0.02 or edge_density > 0.3:
            frame_score += 0.2

        # 4. Noise pattern analysis
        noise_score = _analyze_frame_noise(gray)
        if noise_score > 0.5:
            frame_score += 0.25

        scores.append(min(frame_score, 1.0))

    avg_score = np.mean(scores)

    if avg_score > 0.4:
        indicators.append("frame_level_artifacts_detected")
    if np.std(scores) > 0.3:
        indicators.append("inconsistent_frame_quality")

    return {"score": avg_score, "indicators": indicators}


def _analyze_dct_artifacts(gray: np.ndarray) -> float:
    """Analyze DCT coefficients for generation artifacts."""
    # Resize to standard size for consistent analysis
    h, w = gray.shape
    block_size = 8

    # Analyze 8x8 blocks (JPEG-like)
    block_energies = []

    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = gray[i:i+block_size, j:j+block_size].astype(np.float32)
            dct = cv2.dct(block)
            energy = np.sum(np.abs(dct[1:, 1:]))  # Exclude DC component
            block_energies.append(energy)

    if not block_energies:
        return 0.3

    # Check for unusual uniformity in DCT energies
    cv = np.std(block_energies) / (np.mean(block_energies) + 1e-6)

    if cv < 0.3:  # Too uniform
        return 0.7
    elif cv < 0.5:
        return 0.4
    return 0.2


def _analyze_color_artifacts(hsv: np.ndarray) -> float:
    """Analyze color space for AI artifacts."""
    h, s, v = cv2.split(hsv)

    # Check saturation distribution
    sat_hist = cv2.calcHist([s], [0], None, [256], [0, 256]).flatten()
    sat_hist = sat_hist / sat_hist.sum()

    # Unusual saturation patterns
    sat_entropy = -np.sum(sat_hist[sat_hist > 0] * np.log2(sat_hist[sat_hist > 0]))

    if sat_entropy < 4.0:  # Low entropy suggests artificial colors
        return 0.7
    elif sat_entropy < 5.5:
        return 0.4
    return 0.2


def _analyze_frame_noise(gray: np.ndarray) -> float:
    """Analyze noise patterns in the frame."""
    # Apply Laplacian to detect noise
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    noise_var = laplacian.var()

    # AI images often have very low or very uniform noise
    if noise_var < 50:  # Too smooth
        return 0.7
    elif noise_var > 5000:  # Too noisy (possibly to hide artifacts)
        return 0.5
    return 0.2


def _analyze_temporal_consistency(frames: List[np.ndarray]) -> dict:
    """
    Analyze temporal consistency between frames.

    Deepfakes often have:
    - Flickering artifacts
    - Inconsistent lighting
    - Unnatural motion patterns
    """
    indicators = []
    score = 0.0

    if len(frames) < 3:
        return {"score": 0.5, "indicators": ["insufficient_frames"]}

    # Calculate frame differences
    frame_diffs = []
    for i in range(1, len(frames)):
        prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(prev_gray, curr_gray)
        frame_diffs.append(np.mean(diff))

    # 1. Check for flickering (high variance in frame differences)
    diff_std = np.std(frame_diffs)
    diff_mean = np.mean(frame_diffs)

    if diff_std > diff_mean * 0.8:
        indicators.append("temporal_flickering")
        score += 0.35

    # 2. Check for sudden jumps (inconsistent motion)
    diff_changes = np.diff(frame_diffs)
    sudden_jumps = np.sum(np.abs(diff_changes) > np.mean(frame_diffs) * 2)

    if sudden_jumps > len(frames) * 0.15:
        indicators.append("motion_discontinuities")
        score += 0.3

    # 3. Optical flow analysis for unnatural motion
    flow_scores = []
    for i in range(1, min(len(frames), 10)):
        prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

        # Compute dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        # Calculate flow magnitude
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_scores.append(np.std(mag))

    if flow_scores:
        flow_cv = np.std(flow_scores) / (np.mean(flow_scores) + 1e-6)
        if flow_cv > 1.5:
            indicators.append("unnatural_motion_patterns")
            score += 0.35

    return {"score": min(score, 1.0), "indicators": indicators}


def _analyze_face_manipulation(frames: List[np.ndarray]) -> dict:
    """
    Detect face manipulation indicators (deepfakes).

    Looks for:
    - Face boundary artifacts
    - Blending inconsistencies
    - Unnatural facial features
    """
    indicators = []
    scores = []

    # Load face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    faces_found = 0

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            continue

        faces_found += len(faces)
        frame_score = 0.0

        for (x, y, w, h) in faces:
            # Expand region slightly for boundary analysis
            margin = int(w * 0.1)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)

            face_region = frame[y1:y2, x1:x2]

            if face_region.size == 0:
                continue

            # 1. Check face boundary for blending artifacts
            boundary_score = _analyze_face_boundary(frame, (x, y, w, h))
            if boundary_score > 0.5:
                frame_score += 0.4

            # 2. Check for color inconsistencies
            color_score = _analyze_face_color_consistency(face_region, frame)
            if color_score > 0.5:
                frame_score += 0.3

            # 3. Check texture smoothness (over-smoothed = likely fake)
            texture_score = _analyze_face_texture(face_region)
            if texture_score > 0.5:
                frame_score += 0.3

        if frame_score > 0:
            scores.append(min(frame_score, 1.0))

    if faces_found == 0:
        return {"score": 0.0, "indicators": ["no_faces_detected"]}

    if not scores:
        return {"score": 0.3, "indicators": ["face_analysis_inconclusive"]}

    avg_score = np.mean(scores)

    if avg_score > 0.4:
        indicators.append("potential_face_manipulation")
    if np.std(scores) > 0.25:
        indicators.append("inconsistent_face_quality")

    return {"score": avg_score, "indicators": indicators}


def _analyze_face_boundary(frame: np.ndarray, face_rect: Tuple) -> float:
    """Check for boundary artifacts around face region."""
    x, y, w, h = face_rect

    # Sample boundary pixels
    margin = 5
    inner_boundary = []
    outer_boundary = []

    # Top boundary
    if y - margin > 0:
        outer_boundary.extend(frame[y-margin:y, x:x+w].reshape(-1, 3).tolist())
        inner_boundary.extend(frame[y:y+margin, x:x+w].reshape(-1, 3).tolist())

    if not inner_boundary or not outer_boundary:
        return 0.3

    inner_mean = np.mean(inner_boundary, axis=0)
    outer_mean = np.mean(outer_boundary, axis=0)

    # Large color difference at boundary suggests blending
    color_diff = np.linalg.norm(inner_mean - outer_mean)

    if color_diff > 50:
        return 0.7
    elif color_diff > 30:
        return 0.4
    return 0.2


def _analyze_face_color_consistency(face_region: np.ndarray, full_frame: np.ndarray) -> float:
    """Check if face color is consistent with surrounding."""
    face_hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
    frame_hsv = cv2.cvtColor(full_frame, cv2.COLOR_BGR2HSV)

    face_hue = face_hsv[:, :, 0].mean()
    frame_hue = frame_hsv[:, :, 0].mean()

    # Check if face has significantly different color temperature
    hue_diff = abs(face_hue - frame_hue)

    if hue_diff > 30:
        return 0.7
    elif hue_diff > 15:
        return 0.4
    return 0.2


def _analyze_face_texture(face_region: np.ndarray) -> float:
    """Check for over-smoothed texture (GAN artifact)."""
    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

    # Calculate local variance (texture measure)
    kernel_size = 5
    mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
    sqr_mean = cv2.blur((gray.astype(np.float32))**2, (kernel_size, kernel_size))
    variance = sqr_mean - mean**2

    avg_variance = np.mean(variance)

    # Very low variance suggests over-smoothed (deepfake artifact)
    if avg_variance < 100:
        return 0.8
    elif avg_variance < 300:
        return 0.4
    return 0.2
