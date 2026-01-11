"""
AI-Generated Audio/Voice Detector

Detects synthetic speech and AI-generated audio using:
1. Spectral analysis
2. Prosody analysis
3. Artifact detection
"""
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path


async def detect_ai_audio(file_path: str) -> dict:
    """
    Detect if audio is AI-generated or contains synthetic speech.

    Returns:
        dict with 'is_ai_generated', 'confidence', 'method', and 'details'
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=16000, mono=True)

        # Run multiple detection methods
        spectral_result = _analyze_spectral_features(y, sr)
        prosody_result = _analyze_prosody(y, sr)
        artifact_result = _detect_synthesis_artifacts(y, sr)

        # Combine scores
        weights = {"spectral": 0.35, "prosody": 0.35, "artifacts": 0.30}

        combined_score = (
            spectral_result["score"] * weights["spectral"] +
            prosody_result["score"] * weights["prosody"] +
            artifact_result["score"] * weights["artifacts"]
        )

        is_ai = combined_score > 0.5
        confidence = combined_score * 100

        # Collect all indicators
        indicators = []
        indicators.extend(spectral_result.get("indicators", []))
        indicators.extend(prosody_result.get("indicators", []))
        indicators.extend(artifact_result.get("indicators", []))

        return {
            "is_ai_generated": is_ai,
            "confidence": round(confidence, 2),
            "method": "local_audio_analysis",
            "details": {
                "spectral_score": round(spectral_result["score"] * 100, 2),
                "prosody_score": round(prosody_result["score"] * 100, 2),
                "artifact_score": round(artifact_result["score"] * 100, 2),
                "indicators": indicators,
                "duration_seconds": round(len(y) / sr, 2),
            }
        }

    except Exception as e:
        return {
            "is_ai_generated": None,
            "confidence": 0,
            "method": "error",
            "error": str(e),
            "details": {"message": "Failed to analyze audio file"}
        }


def _analyze_spectral_features(y: np.ndarray, sr: int) -> dict:
    """
    Analyze spectral characteristics for AI synthesis markers.

    AI-generated audio often has:
    - Unusual spectral smoothness
    - Missing or artificial harmonics
    - Unnatural frequency transitions
    """
    indicators = []
    score = 0.0

    # Compute spectrogram
    S = np.abs(librosa.stft(y))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    # 1. Spectral centroid analysis
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_std = np.std(spectral_centroids)

    # Very low variation suggests synthetic audio
    if centroid_std < 200:
        indicators.append("unnaturally_stable_spectral_centroid")
        score += 0.3

    # 2. Spectral bandwidth analysis
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    bw_variation = np.std(spec_bw) / (np.mean(spec_bw) + 1e-6)

    if bw_variation < 0.15:
        indicators.append("low_bandwidth_variation")
        score += 0.25

    # 3. Spectral rolloff - check for artificial cutoffs
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    rolloff_mean = np.mean(rolloff)

    # Unusual high-frequency cutoff patterns
    if rolloff_mean < 2000 or rolloff_mean > 7000:
        indicators.append("unusual_frequency_rolloff")
        score += 0.2

    # 4. Harmonic-to-noise ratio
    # AI voices sometimes have unnatural H2N ratios
    harmonic, percussive = librosa.effects.hpss(y)
    h2n = np.sum(harmonic**2) / (np.sum(percussive**2) + 1e-6)

    if h2n > 100:  # Unusually clean/synthetic
        indicators.append("unusually_high_harmonic_ratio")
        score += 0.25

    return {"score": min(score, 1.0), "indicators": indicators}


def _analyze_prosody(y: np.ndarray, sr: int) -> dict:
    """
    Analyze prosody (pitch, rhythm, intonation) for naturalness.

    AI-generated speech often has:
    - Unnatural pitch patterns
    - Robotic rhythm
    - Inconsistent speaking rate
    """
    indicators = []
    score = 0.0

    # Extract pitch (F0)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr
    )

    # Remove NaN values
    f0_valid = f0[~np.isnan(f0)]

    if len(f0_valid) < 10:
        return {"score": 0.5, "indicators": ["insufficient_voiced_segments"]}

    # 1. Pitch variation analysis
    pitch_std = np.std(f0_valid)
    pitch_mean = np.mean(f0_valid)
    pitch_cv = pitch_std / (pitch_mean + 1e-6)

    # Too regular pitch suggests AI
    if pitch_cv < 0.08:
        indicators.append("monotonic_pitch")
        score += 0.35
    elif pitch_cv > 0.5:
        indicators.append("erratic_pitch")
        score += 0.2

    # 2. Pitch contour smoothness
    pitch_diff = np.diff(f0_valid)
    pitch_jitter = np.std(pitch_diff)

    # Very smooth pitch changes (AI) or very jittery (also AI artifacts)
    if pitch_jitter < 2:
        indicators.append("unnaturally_smooth_pitch_transitions")
        score += 0.3
    elif pitch_jitter > 50:
        indicators.append("pitch_discontinuities")
        score += 0.25

    # 3. Rhythm analysis using onset detection
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    if len(onset_frames) > 2:
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        intervals = np.diff(onset_times)

        if len(intervals) > 1:
            interval_cv = np.std(intervals) / (np.mean(intervals) + 1e-6)

            # Too regular rhythm
            if interval_cv < 0.2:
                indicators.append("robotic_rhythm")
                score += 0.25

    return {"score": min(score, 1.0), "indicators": indicators}


def _detect_synthesis_artifacts(y: np.ndarray, sr: int) -> dict:
    """
    Detect specific artifacts common in AI-synthesized audio.

    Looks for:
    - Vocoder artifacts
    - Concatenation glitches
    - Phase discontinuities
    - Unusual silence patterns
    """
    indicators = []
    score = 0.0

    # 1. Check for vocoder-like phase patterns
    S = librosa.stft(y)
    phase = np.angle(S)
    phase_coherence = np.mean(np.abs(np.diff(phase, axis=1)))

    # Very coherent phase suggests vocoder synthesis
    if phase_coherence < 0.5:
        indicators.append("vocoder_phase_patterns")
        score += 0.35

    # 2. Check for unnatural silence/pause patterns
    rms = librosa.feature.rms(y=y)[0]
    silence_threshold = 0.01
    is_silent = rms < silence_threshold

    # Count silence transitions
    silence_transitions = np.sum(np.diff(is_silent.astype(int)) != 0)
    audio_duration = len(y) / sr

    transitions_per_second = silence_transitions / (audio_duration + 1e-6)

    # Unusual transition patterns
    if transitions_per_second > 5:
        indicators.append("unusual_silence_patterns")
        score += 0.2

    # 3. Check for repetitive patterns (copy-paste artifacts)
    # Use autocorrelation
    autocorr = np.correlate(y[:min(len(y), sr*2)], y[:min(len(y), sr*2)], mode='same')
    autocorr = autocorr / (np.max(autocorr) + 1e-6)

    # Look for suspicious periodic peaks
    peaks = []
    for i in range(len(autocorr)//4, len(autocorr)//2):
        if autocorr[i] > 0.5:
            peaks.append(i)

    if len(peaks) > 3:
        indicators.append("repetitive_audio_patterns")
        score += 0.25

    # 4. Mel-frequency analysis for synthesis artifacts
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Check MFCC variance across time
    mfcc_time_var = np.var(mfccs, axis=1)

    # Very low variance in higher MFCCs suggests synthetic audio
    if np.mean(mfcc_time_var[5:]) < 5:
        indicators.append("synthetic_mfcc_patterns")
        score += 0.3

    return {"score": min(score, 1.0), "indicators": indicators}
