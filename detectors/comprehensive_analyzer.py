"""
Comprehensive Multi-Technique Analyzer

Combines multiple detection techniques for higher accuracy:
- Each technique provides independent evidence
- Results are aggregated with weighted scoring
- Agreement between techniques increases confidence
- Disagreement flags uncertainty
"""

import asyncio
from typing import Dict, List, Any
import numpy as np

from .fourier_analyzer import analyze_image_fft, analyze_audio_fft, analyze_video_fft
from .learning_analyzers import (
    analyze_wavelet, analyze_ela, analyze_noise, analyze_histogram,
    analyze_gradient, analyze_autocorrelation, analyze_optical_flow, analyze_cepstral
)


# Technique weights based on reliability for each media type
IMAGE_TECHNIQUES = {
    "fourier": {"weight": 0.15, "name": "Fourier Transform"},
    "wavelet": {"weight": 0.15, "name": "Wavelet Transform"},
    "ela": {"weight": 0.12, "name": "Error Level Analysis"},
    "noise": {"weight": 0.18, "name": "Noise Analysis"},
    "histogram": {"weight": 0.10, "name": "Histogram Analysis"},
    "gradient": {"weight": 0.15, "name": "Gradient & Edge"},
    "autocorrelation": {"weight": 0.15, "name": "Autocorrelation"},
}

AUDIO_TECHNIQUES = {
    "fourier": {"weight": 0.25, "name": "Fourier Transform"},
    "wavelet": {"weight": 0.20, "name": "Wavelet Transform"},
    "autocorrelation": {"weight": 0.25, "name": "Autocorrelation"},
    "cepstral": {"weight": 0.30, "name": "Cepstral Analysis (MFCCs)"},
}

VIDEO_TECHNIQUES = {
    "fourier": {"weight": 0.35, "name": "Fourier Transform"},
    "optical_flow": {"weight": 0.65, "name": "Optical Flow"},
}


async def comprehensive_image_analysis(file_path: str) -> Dict[str, Any]:
    """
    Run all image analysis techniques and aggregate results.
    """
    results = {}
    errors = []

    # Run all techniques concurrently
    tasks = {
        "fourier": analyze_image_fft(file_path),
        "wavelet": analyze_wavelet(file_path, "image"),
        "ela": analyze_ela(file_path),
        "noise": analyze_noise(file_path),
        "histogram": analyze_histogram(file_path),
        "gradient": analyze_gradient(file_path),
        "autocorrelation": analyze_autocorrelation(file_path, "image"),
    }

    # Gather results
    gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)

    for technique, result in zip(tasks.keys(), gathered):
        if isinstance(result, Exception):
            errors.append({"technique": technique, "error": str(result)})
            results[technique] = None
        else:
            results[technique] = result

    # Aggregate scores
    aggregation = _aggregate_results(results, IMAGE_TECHNIQUES)

    return {
        "media_type": "image",
        "techniques_run": len(tasks),
        "techniques_succeeded": len(tasks) - len(errors),
        "individual_results": results,
        "aggregation": aggregation,
        "errors": errors if errors else None,
    }


async def comprehensive_audio_analysis(file_path: str) -> Dict[str, Any]:
    """
    Run all audio analysis techniques and aggregate results.
    """
    results = {}
    errors = []

    tasks = {
        "fourier": analyze_audio_fft(file_path),
        "wavelet": analyze_wavelet(file_path, "audio"),
        "autocorrelation": analyze_autocorrelation(file_path, "audio"),
        "cepstral": analyze_cepstral(file_path),
    }

    gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)

    for technique, result in zip(tasks.keys(), gathered):
        if isinstance(result, Exception):
            errors.append({"technique": technique, "error": str(result)})
            results[technique] = None
        else:
            results[technique] = result

    aggregation = _aggregate_results(results, AUDIO_TECHNIQUES)

    return {
        "media_type": "audio",
        "techniques_run": len(tasks),
        "techniques_succeeded": len(tasks) - len(errors),
        "individual_results": results,
        "aggregation": aggregation,
        "errors": errors if errors else None,
    }


async def comprehensive_video_analysis(file_path: str) -> Dict[str, Any]:
    """
    Run all video analysis techniques and aggregate results.
    """
    results = {}
    errors = []

    tasks = {
        "fourier": analyze_video_fft(file_path),
        "optical_flow": analyze_optical_flow(file_path),
    }

    gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)

    for technique, result in zip(tasks.keys(), gathered):
        if isinstance(result, Exception):
            errors.append({"technique": technique, "error": str(result)})
            results[technique] = None
        else:
            results[technique] = result

    aggregation = _aggregate_results(results, VIDEO_TECHNIQUES)

    return {
        "media_type": "video",
        "techniques_run": len(tasks),
        "techniques_succeeded": len(tasks) - len(errors),
        "individual_results": results,
        "aggregation": aggregation,
        "errors": errors if errors else None,
    }


def _aggregate_results(results: Dict, technique_config: Dict) -> Dict[str, Any]:
    """
    Aggregate individual technique results into a combined score.

    Uses:
    1. Weighted average of individual scores
    2. Agreement analysis (do techniques agree?)
    3. Confidence adjustment based on agreement
    """
    scores = []
    weights = []
    technique_scores = {}
    all_indicators = []

    for technique, config in technique_config.items():
        result = results.get(technique)
        if result is None:
            continue

        # Extract AI likelihood score
        score = None
        if "analysis" in result and result["analysis"]:
            score = result["analysis"].get("ai_likelihood_score", 0)
        elif "ai_likelihood_score" in result:
            score = result.get("ai_likelihood_score", 0)

        if score is not None:
            scores.append(score)
            weights.append(config["weight"])
            technique_scores[technique] = {
                "name": config["name"],
                "score": score,
                "weight": config["weight"],
            }

            # Collect indicators
            if "analysis" in result and result["analysis"]:
                indicators = result["analysis"].get("ai_indicators", [])
                for ind in indicators:
                    all_indicators.append({
                        "technique": config["name"],
                        **ind
                    })

    if not scores:
        return {
            "combined_score": 0,
            "confidence": "low",
            "verdict": "Unable to analyze",
            "technique_scores": {},
            "agreement": None,
        }

    # Calculate weighted average
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize
    scores = np.array(scores)

    weighted_score = np.sum(scores * weights)

    # Calculate agreement (how much techniques agree)
    score_std = np.std(scores)
    score_range = np.max(scores) - np.min(scores)

    # Agreement score: lower std = higher agreement
    if len(scores) > 1:
        agreement_score = max(0, 100 - score_std * 2)
    else:
        agreement_score = 50  # Single technique = uncertain agreement

    # Classify techniques by their verdict
    ai_votes = sum(1 for s in scores if s >= 50)
    human_votes = sum(1 for s in scores if s < 50)

    # Determine confidence level
    if agreement_score >= 80 and len(scores) >= 3:
        confidence = "high"
    elif agreement_score >= 60 and len(scores) >= 2:
        confidence = "medium"
    else:
        confidence = "low"

    # Adjust combined score based on agreement
    # Strong agreement boosts confidence in the direction of consensus
    if agreement_score >= 80:
        # Amplify the signal when techniques agree
        if weighted_score >= 50:
            weighted_score = min(95, weighted_score + (100 - weighted_score) * 0.2)
        else:
            weighted_score = max(5, weighted_score - weighted_score * 0.2)

    # Determine verdict
    if weighted_score >= 70:
        verdict = "Likely AI-Generated"
        verdict_class = "ai"
    elif weighted_score >= 50:
        verdict = "Possibly AI-Generated"
        verdict_class = "uncertain-ai"
    elif weighted_score >= 30:
        verdict = "Possibly Authentic"
        verdict_class = "uncertain-human"
    else:
        verdict = "Likely Authentic"
        verdict_class = "human"

    # Group indicators by severity
    high_severity = [i for i in all_indicators if i.get("severity") == "high"]
    medium_severity = [i for i in all_indicators if i.get("severity") == "medium"]
    low_severity = [i for i in all_indicators if i.get("severity") == "low"]

    return {
        "combined_score": round(float(weighted_score), 1),
        "confidence": confidence,
        "verdict": verdict,
        "verdict_class": verdict_class,
        "agreement": {
            "score": round(float(agreement_score), 1),
            "description": _get_agreement_description(agreement_score, ai_votes, human_votes),
            "ai_votes": ai_votes,
            "human_votes": human_votes,
            "score_range": round(float(score_range), 1),
        },
        "technique_scores": technique_scores,
        "indicators_summary": {
            "high_severity": len(high_severity),
            "medium_severity": len(medium_severity),
            "low_severity": len(low_severity),
            "total": len(all_indicators),
        },
        "all_indicators": all_indicators,
    }


def _get_agreement_description(agreement_score: float, ai_votes: int, human_votes: int) -> str:
    """Generate human-readable agreement description."""
    total = ai_votes + human_votes

    if agreement_score >= 80:
        if ai_votes == total:
            return f"Strong consensus: All {total} techniques indicate AI generation"
        elif human_votes == total:
            return f"Strong consensus: All {total} techniques indicate authentic content"
        else:
            return f"High agreement among techniques ({agreement_score:.0f}%)"
    elif agreement_score >= 60:
        return f"Moderate agreement: {ai_votes} techniques suggest AI, {human_votes} suggest authentic"
    else:
        return f"Low agreement: Techniques show conflicting results (range: {ai_votes} AI vs {human_votes} authentic)"
