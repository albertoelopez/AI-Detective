"""
Fourier Transform Analysis for AI Detection - Educational Module

This module demonstrates how Fourier/frequency domain analysis can reveal
artifacts in AI-generated content.

Key concepts:
- FFT (Fast Fourier Transform) converts signals from time/space domain to frequency domain
- AI-generated content often has telltale frequency patterns:
  - Images: Grid artifacts, missing high frequencies, periodic patterns from upsampling
  - Audio: Unnatural harmonics, vocoder artifacts, frequency gaps
  - Video: Temporal frequency inconsistencies between frames
"""

import numpy as np
from PIL import Image
import cv2
import librosa
import base64
from io import BytesIO
from typing import Tuple
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


async def analyze_image_fft(file_path: str) -> dict:
    """
    Perform 2D FFT analysis on an image.

    What to look for in AI-generated images:
    1. Grid patterns in frequency domain (from convolution layers)
    2. Unusual peaks at specific frequencies (upsampling artifacts)
    3. Missing high-frequency detail (over-smoothing)
    4. Periodic patterns (GAN checkerboard artifacts)
    """
    # Load and convert to grayscale
    img = Image.open(file_path).convert('L')
    img_array = np.array(img, dtype=np.float32)

    # Apply 2D FFT
    f_transform = np.fft.fft2(img_array)
    f_shift = np.fft.fftshift(f_transform)  # Shift zero frequency to center
    magnitude_spectrum = np.log1p(np.abs(f_shift))  # Log scale for visibility

    # Analyze frequency characteristics
    analysis = _analyze_image_spectrum(magnitude_spectrum, img_array)

    # Generate visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(img_array, cmap='gray')
    axes[0].set_title('Original Image (Spatial Domain)')
    axes[0].axis('off')

    # Magnitude spectrum
    im = axes[1].imshow(magnitude_spectrum, cmap='hot')
    axes[1].set_title('FFT Magnitude Spectrum (Frequency Domain)')
    axes[1].set_xlabel('Horizontal Frequency')
    axes[1].set_ylabel('Vertical Frequency')
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Annotated spectrum with regions
    axes[2].imshow(magnitude_spectrum, cmap='hot')
    axes[2].set_title('Frequency Regions Explained')
    _add_frequency_annotations(axes[2], magnitude_spectrum.shape)

    plt.tight_layout()

    # Convert plot to base64
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return {
        "visualization": plot_base64,
        "analysis": analysis,
        "explanation": _get_image_fft_explanation(),
    }


def _analyze_image_spectrum(spectrum: np.ndarray, original: np.ndarray) -> dict:
    """Analyze the frequency spectrum for AI indicators."""
    h, w = spectrum.shape
    center_y, center_x = h // 2, w // 2

    # Create radial distance map
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    # Analyze different frequency bands
    low_freq_mask = r < min(h, w) * 0.1
    mid_freq_mask = (r >= min(h, w) * 0.1) & (r < min(h, w) * 0.3)
    high_freq_mask = r >= min(h, w) * 0.3

    low_energy = np.mean(spectrum[low_freq_mask])
    mid_energy = np.mean(spectrum[mid_freq_mask])
    high_energy = np.mean(spectrum[high_freq_mask])

    total_energy = low_energy + mid_energy + high_energy

    indicators = []
    ai_score = 0.0

    # Check 1: High frequency content (AI often lacks fine detail)
    high_ratio = high_energy / total_energy if total_energy > 0 else 0
    if high_ratio < 0.15:
        indicators.append({
            "name": "Low high-frequency content",
            "description": "AI images often lack fine detail visible in high frequencies",
            "severity": "medium"
        })
        ai_score += 0.3

    # Check 2: Look for periodic artifacts (peaks at regular intervals)
    peak_score = _detect_periodic_peaks(spectrum)
    if peak_score > 0.5:
        indicators.append({
            "name": "Periodic frequency peaks",
            "description": "Regular patterns may indicate GAN upsampling artifacts",
            "severity": "high"
        })
        ai_score += 0.35

    # Check 3: Analyze radial symmetry (natural images have more variation)
    symmetry_score = _analyze_radial_symmetry(spectrum)
    if symmetry_score > 0.8:
        indicators.append({
            "name": "Unusual radial symmetry",
            "description": "AI-generated images sometimes show unnatural frequency symmetry",
            "severity": "low"
        })
        ai_score += 0.15

    return {
        "frequency_distribution": {
            "low_frequency_percent": round(low_energy / total_energy * 100, 1) if total_energy > 0 else 0,
            "mid_frequency_percent": round(mid_energy / total_energy * 100, 1) if total_energy > 0 else 0,
            "high_frequency_percent": round(high_ratio * 100, 1),
        },
        "ai_indicators": indicators,
        "ai_likelihood_score": round(min(ai_score, 1.0) * 100, 1),
    }


def _detect_periodic_peaks(spectrum: np.ndarray) -> float:
    """Detect periodic peaks that might indicate GAN artifacts."""
    h, w = spectrum.shape
    center_y, center_x = h // 2, w // 2

    # Sample along radial lines
    angles = np.linspace(0, 2 * np.pi, 36)
    radii = np.linspace(10, min(h, w) // 3, 50)

    profiles = []
    for angle in angles:
        profile = []
        for r in radii:
            y = int(center_y + r * np.sin(angle))
            x = int(center_x + r * np.cos(angle))
            if 0 <= y < h and 0 <= x < w:
                profile.append(spectrum[y, x])
        if profile:
            profiles.append(profile)

    if not profiles:
        return 0.0

    # Check for periodic patterns in profiles
    peak_scores = []
    for profile in profiles:
        if len(profile) > 10:
            profile = np.array(profile)
            # FFT of the radial profile to find periodicity
            profile_fft = np.abs(np.fft.fft(profile - np.mean(profile)))
            # Look for strong peaks (excluding DC)
            peak_ratio = np.max(profile_fft[2:len(profile_fft)//2]) / (np.mean(profile_fft[2:]) + 1e-6)
            peak_scores.append(min(peak_ratio / 5, 1.0))

    return np.mean(peak_scores) if peak_scores else 0.0


def _analyze_radial_symmetry(spectrum: np.ndarray) -> float:
    """Analyze how radially symmetric the spectrum is."""
    h, w = spectrum.shape
    center_y, center_x = h // 2, w // 2

    # Compare opposite quadrants
    q1 = spectrum[:center_y, :center_x]
    q3 = spectrum[center_y:, center_x:][::-1, ::-1]

    min_h = min(q1.shape[0], q3.shape[0])
    min_w = min(q1.shape[1], q3.shape[1])

    q1 = q1[:min_h, :min_w]
    q3 = q3[:min_h, :min_w]

    correlation = np.corrcoef(q1.flatten(), q3.flatten())[0, 1]
    return abs(correlation) if not np.isnan(correlation) else 0.5


def _add_frequency_annotations(ax, shape):
    """Add educational annotations to the frequency plot."""
    h, w = shape
    center_y, center_x = h // 2, w // 2

    # Draw circles for frequency bands
    for radius, label, color in [
        (min(h, w) * 0.1, 'Low Freq\n(overall shape)', 'cyan'),
        (min(h, w) * 0.3, 'Mid Freq\n(textures)', 'yellow'),
    ]:
        circle = plt.Circle((center_x, center_y), radius, fill=False,
                           color=color, linewidth=2, linestyle='--')
        ax.add_patch(circle)

    # Add text annotations
    ax.annotate('CENTER = Low frequencies\n(image brightness, large shapes)',
                xy=(center_x, center_y), xytext=(center_x + w*0.25, center_y - h*0.3),
                fontsize=8, color='white', backgroundcolor='black',
                arrowprops=dict(arrowstyle='->', color='white'))

    ax.annotate('EDGES = High frequencies\n(fine details, edges, noise)',
                xy=(w*0.9, h*0.1), xytext=(w*0.6, h*0.15),
                fontsize=8, color='white', backgroundcolor='black',
                arrowprops=dict(arrowstyle='->', color='white'))

    ax.axis('off')


def _get_image_fft_explanation() -> dict:
    return {
        "title": "Understanding Image FFT for AI Detection",
        "sections": [
            {
                "heading": "What is FFT?",
                "content": "The Fast Fourier Transform converts an image from spatial domain (pixels) to frequency domain. Each point in the FFT represents how much of a particular frequency pattern exists in the image."
            },
            {
                "heading": "Reading the Spectrum",
                "content": "CENTER = Low frequencies (overall brightness, large shapes). EDGES = High frequencies (fine details, sharp edges, noise). Brighter areas = stronger presence of that frequency."
            },
            {
                "heading": "AI Detection Clues",
                "content": "1) Missing high frequencies: AI images often look 'too smooth' - the edges of the FFT are darker than natural photos. 2) Grid patterns: GAN upsampling can create periodic artifacts visible as bright spots at regular intervals. 3) Unusual symmetry: Some AI models produce unnaturally symmetric frequency patterns."
            },
            {
                "heading": "Limitations",
                "content": "FFT analysis alone isn't definitive. JPEG compression, resizing, and post-processing affect frequencies. Use this as one tool among many."
            }
        ]
    }


async def analyze_audio_fft(file_path: str) -> dict:
    """
    Perform FFT/spectrogram analysis on audio.

    What to look for in AI-generated audio:
    1. Unnatural harmonic structure
    2. Missing or artificial overtones
    3. Vocoder artifacts (vertical lines in spectrogram)
    4. Frequency gaps or unusual cutoffs
    """
    # Load audio
    y, sr = librosa.load(file_path, sr=22050, mono=True, duration=30)

    # Compute spectrogram (STFT)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Compute mel spectrogram (more perceptually relevant)
    S_mel = librosa.feature.melspectrogram(y=y, sr=sr)
    S_mel_db = librosa.amplitude_to_db(S_mel, ref=np.max)

    # Analyze
    analysis = _analyze_audio_spectrum(y, sr, S_db)

    # Generate visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Waveform
    times = np.arange(len(y)) / sr
    axes[0, 0].plot(times, y, linewidth=0.5)
    axes[0, 0].set_title('Waveform (Time Domain)')
    axes[0, 0].set_xlabel('Time (seconds)')
    axes[0, 0].set_ylabel('Amplitude')

    # FFT of full signal
    fft_full = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1/sr)
    axes[0, 1].semilogy(freqs[:len(freqs)//4], fft_full[:len(freqs)//4])
    axes[0, 1].set_title('FFT Magnitude (Frequency Domain)')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Magnitude (log scale)')
    axes[0, 1].set_xlim(0, sr/8)

    # Spectrogram
    img1 = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=axes[1, 0])
    axes[1, 0].set_title('Spectrogram (STFT)')
    axes[1, 0].set_ylabel('Frequency (Hz)')
    plt.colorbar(img1, ax=axes[1, 0], format='%+2.0f dB')

    # Mel spectrogram
    img2 = librosa.display.specshow(S_mel_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[1, 1])
    axes[1, 1].set_title('Mel Spectrogram (Perceptual Frequencies)')
    axes[1, 1].set_ylabel('Mel Frequency')
    plt.colorbar(img2, ax=axes[1, 1], format='%+2.0f dB')

    plt.tight_layout()

    # Convert to base64
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return {
        "visualization": plot_base64,
        "analysis": analysis,
        "explanation": _get_audio_fft_explanation(),
    }


def _analyze_audio_spectrum(y: np.ndarray, sr: int, S_db: np.ndarray) -> dict:
    """Analyze audio spectrum for AI indicators."""
    indicators = []
    ai_score = 0.0

    # 1. Analyze harmonic structure
    harmonic, percussive = librosa.effects.hpss(y)
    harmonic_ratio = np.sum(harmonic**2) / (np.sum(y**2) + 1e-10)

    if harmonic_ratio > 0.95:
        indicators.append({
            "name": "Unusually high harmonic content",
            "description": "Natural speech/music has more percussive elements. Very clean harmonics may indicate synthesis.",
            "severity": "medium"
        })
        ai_score += 0.25

    # 2. Check for frequency cutoffs
    fft_full = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1/sr)

    # Check for sharp cutoff
    high_freq_idx = freqs > 8000
    high_freq_energy = np.mean(fft_full[high_freq_idx]) if np.any(high_freq_idx) else 0
    total_energy = np.mean(fft_full)

    if high_freq_energy < total_energy * 0.01:
        indicators.append({
            "name": "Sharp high-frequency cutoff",
            "description": "Audio may have been generated at lower sample rate or has vocoder artifacts",
            "severity": "medium"
        })
        ai_score += 0.2

    # 3. Check spectrogram for vertical lines (vocoder artifacts)
    spec_diff = np.diff(S_db, axis=1)
    sudden_changes = np.sum(np.abs(spec_diff) > 20) / spec_diff.size

    if sudden_changes > 0.05:
        indicators.append({
            "name": "Spectral discontinuities",
            "description": "Sudden frequency changes may indicate frame-based synthesis",
            "severity": "high"
        })
        ai_score += 0.3

    # 4. Analyze pitch stability
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
    f0_valid = f0[~np.isnan(f0)]

    if len(f0_valid) > 10:
        pitch_std = np.std(f0_valid)
        if pitch_std < 5:
            indicators.append({
                "name": "Unnaturally stable pitch",
                "description": "Human voice naturally varies in pitch. Very stable pitch suggests synthesis.",
                "severity": "medium"
            })
            ai_score += 0.25

    return {
        "audio_stats": {
            "duration_seconds": round(len(y) / sr, 2),
            "sample_rate": sr,
            "harmonic_ratio": round(harmonic_ratio * 100, 1),
        },
        "ai_indicators": indicators,
        "ai_likelihood_score": round(min(ai_score, 1.0) * 100, 1),
    }


def _get_audio_fft_explanation() -> dict:
    return {
        "title": "Understanding Audio FFT for AI Detection",
        "sections": [
            {
                "heading": "Time vs Frequency Domain",
                "content": "The waveform shows amplitude over time. FFT reveals which frequencies are present. The spectrogram shows how frequencies change over time - it's like a movie of the FFT."
            },
            {
                "heading": "Reading a Spectrogram",
                "content": "X-axis = time, Y-axis = frequency, Color = intensity. Horizontal lines = sustained tones. Vertical lines = sudden changes (clicks, consonants). Harmonics appear as parallel horizontal bands."
            },
            {
                "heading": "AI Detection Clues",
                "content": "1) Vertical stripes: Vocoder-based synthesis processes audio in frames, causing visible boundaries. 2) Missing frequencies: AI may not reproduce full frequency range. 3) Too-perfect harmonics: Natural audio has slight variations; AI can be too clean. 4) Flat pitch: Human voice naturally wobbles; synthetic voices may be too stable."
            },
            {
                "heading": "Mel Spectrogram",
                "content": "The mel scale matches human hearing perception - we're more sensitive to low frequency differences. This view often reveals artifacts more clearly for speech analysis."
            }
        ]
    }


async def analyze_video_fft(file_path: str) -> dict:
    """
    Perform temporal and spatial FFT analysis on video.

    Analyzes:
    1. Spatial FFT of individual frames (like image analysis)
    2. Temporal FFT across frames (detecting temporal artifacts)
    """
    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        return {"error": "Could not open video file"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample frames evenly
    sample_count = min(60, frame_count)
    frame_indices = np.linspace(0, frame_count - 1, sample_count, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Resize for consistent analysis
            gray = cv2.resize(gray, (256, 256))
            frames.append(gray)

    cap.release()

    if len(frames) < 10:
        return {"error": "Not enough frames to analyze"}

    frames = np.array(frames)

    # Spatial FFT analysis (average across frames)
    spatial_ffts = []
    for frame in frames:
        f = np.fft.fft2(frame)
        f_shift = np.fft.fftshift(f)
        spatial_ffts.append(np.log1p(np.abs(f_shift)))

    avg_spatial_fft = np.mean(spatial_ffts, axis=0)

    # Temporal FFT (how each pixel changes over time)
    temporal_fft = np.fft.fft(frames, axis=0)
    temporal_magnitude = np.mean(np.abs(temporal_fft), axis=(1, 2))

    # Analysis
    analysis = _analyze_video_spectrum(frames, avg_spatial_fft, temporal_magnitude, fps)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Sample frame
    axes[0, 0].imshow(frames[len(frames)//2], cmap='gray')
    axes[0, 0].set_title('Sample Frame (Spatial Domain)')
    axes[0, 0].axis('off')

    # Average spatial FFT
    im1 = axes[0, 1].imshow(avg_spatial_fft, cmap='hot')
    axes[0, 1].set_title('Average Spatial FFT Across Frames')
    axes[0, 1].set_xlabel('Horizontal Frequency')
    axes[0, 1].set_ylabel('Vertical Frequency')
    plt.colorbar(im1, ax=axes[0, 1])

    # Temporal FFT
    temporal_freqs = np.fft.fftfreq(len(frames), 1/fps)[:len(frames)//2]
    axes[1, 0].plot(temporal_freqs, temporal_magnitude[:len(frames)//2])
    axes[1, 0].set_title('Temporal FFT (How Pixels Change Over Time)')
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Magnitude')
    axes[1, 0].axvline(x=fps/2, color='r', linestyle='--', label=f'Nyquist ({fps/2:.1f} Hz)')
    axes[1, 0].legend()

    # Frame-to-frame difference magnitude over time
    frame_diffs = np.diff(frames.astype(np.float32), axis=0)
    diff_magnitudes = np.mean(np.abs(frame_diffs), axis=(1, 2))
    time_axis = np.arange(len(diff_magnitudes)) / fps

    axes[1, 1].plot(time_axis, diff_magnitudes)
    axes[1, 1].set_title('Frame-to-Frame Difference (Motion/Change)')
    axes[1, 1].set_xlabel('Time (seconds)')
    axes[1, 1].set_ylabel('Average Pixel Difference')

    plt.tight_layout()

    # Convert to base64
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return {
        "visualization": plot_base64,
        "analysis": analysis,
        "explanation": _get_video_fft_explanation(),
    }


def _analyze_video_spectrum(frames: np.ndarray, spatial_fft: np.ndarray,
                           temporal_mag: np.ndarray, fps: float) -> dict:
    """Analyze video frequency characteristics for AI indicators."""
    indicators = []
    ai_score = 0.0

    # 1. Spatial consistency check
    frame_ffts = [np.fft.fftshift(np.fft.fft2(f)) for f in frames]
    fft_similarities = []
    for i in range(1, len(frame_ffts)):
        corr = np.corrcoef(
            np.abs(frame_ffts[i]).flatten(),
            np.abs(frame_ffts[i-1]).flatten()
        )[0, 1]
        fft_similarities.append(corr)

    avg_similarity = np.mean(fft_similarities)
    if avg_similarity > 0.98:
        indicators.append({
            "name": "Unusually consistent spatial frequencies",
            "description": "Frame-to-frame frequency content is too similar, may indicate synthetic generation",
            "severity": "medium"
        })
        ai_score += 0.25

    # 2. Temporal frequency analysis
    # Check for unusual periodic patterns
    temporal_fft_of_mag = np.abs(np.fft.fft(temporal_mag - np.mean(temporal_mag)))
    peak_ratio = np.max(temporal_fft_of_mag[2:len(temporal_fft_of_mag)//2]) / (np.mean(temporal_fft_of_mag[2:]) + 1e-6)

    if peak_ratio > 10:
        indicators.append({
            "name": "Strong periodic temporal pattern",
            "description": "Unusual rhythmic changes may indicate frame interpolation or looping",
            "severity": "high"
        })
        ai_score += 0.35

    # 3. Frame difference analysis
    frame_diffs = np.diff(frames.astype(np.float32), axis=0)
    diff_magnitudes = np.mean(np.abs(frame_diffs), axis=(1, 2))
    diff_cv = np.std(diff_magnitudes) / (np.mean(diff_magnitudes) + 1e-6)

    if diff_cv < 0.2:
        indicators.append({
            "name": "Unnaturally smooth motion",
            "description": "Frame differences are too consistent, may indicate AI interpolation",
            "severity": "medium"
        })
        ai_score += 0.2

    return {
        "video_stats": {
            "frames_analyzed": len(frames),
            "fps": round(fps, 2),
            "spatial_consistency": round(avg_similarity * 100, 1),
        },
        "ai_indicators": indicators,
        "ai_likelihood_score": round(min(ai_score, 1.0) * 100, 1),
    }


def _get_video_fft_explanation() -> dict:
    return {
        "title": "Understanding Video FFT for AI Detection",
        "sections": [
            {
                "heading": "Two Types of Analysis",
                "content": "SPATIAL FFT: Analyzes each frame like an image (what patterns exist). TEMPORAL FFT: Analyzes how pixels change over time (motion patterns)."
            },
            {
                "heading": "Spatial FFT (Per-Frame)",
                "content": "Same as image FFT - looking for artifacts in individual frames. We average across frames to see consistent patterns. AI-generated videos may have the same artifacts in every frame."
            },
            {
                "heading": "Temporal FFT",
                "content": "Treats each pixel as a signal over time. Reveals: flickering (high temporal frequency), scene changes (spikes), unnatural motion (unusual patterns). Deepfakes often have temporal inconsistencies around manipulated regions."
            },
            {
                "heading": "AI Detection Clues",
                "content": "1) Too-consistent frame FFTs: Real video has natural variation. 2) Periodic temporal patterns: May indicate frame interpolation or looping. 3) Smooth motion: AI interpolation can create unnaturally smooth transitions. 4) Temporal flickering: Some deepfakes show high-frequency temporal artifacts, especially around faces."
            }
        ]
    }
