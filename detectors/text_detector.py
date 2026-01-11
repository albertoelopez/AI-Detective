"""
AI-Generated Text Detector

IMPORTANT DISCLAIMER:
AI text detection is highly unreliable and should NOT be used for:
- Academic integrity decisions
- Employment decisions
- Legal matters
- Any high-stakes determinations

This detector uses statistical analysis and should be treated as
an indicator, not definitive proof.

Detection methods:
1. Perplexity analysis
2. Burstiness measurement
3. Statistical patterns
4. Vocabulary analysis
"""
import re
import math
from collections import Counter
from typing import List


async def detect_ai_text(text: str) -> dict:
    """
    Analyze text for AI-generation indicators.

    WARNING: AI text detection has high false positive and negative rates.
    Results should be interpreted with extreme caution.

    Returns:
        dict with 'is_ai_generated', 'confidence', 'method', and 'details'
    """
    if len(text.strip()) < 100:
        return {
            "is_ai_generated": None,
            "confidence": 0,
            "method": "insufficient_text",
            "error": "Text too short for reliable analysis (need 100+ characters)",
            "disclaimer": _get_disclaimer(),
        }

    # Run analysis methods
    perplexity_result = _analyze_perplexity_proxy(text)
    burstiness_result = _analyze_burstiness(text)
    vocabulary_result = _analyze_vocabulary(text)
    pattern_result = _analyze_patterns(text)

    # Combine scores with weights
    weights = {
        "perplexity": 0.25,
        "burstiness": 0.30,
        "vocabulary": 0.25,
        "patterns": 0.20,
    }

    combined_score = (
        perplexity_result["score"] * weights["perplexity"] +
        burstiness_result["score"] * weights["burstiness"] +
        vocabulary_result["score"] * weights["vocabulary"] +
        pattern_result["score"] * weights["patterns"]
    )

    # Be conservative - only flag as AI if strong signals
    is_ai = combined_score > 0.55
    # Cap confidence to reflect inherent uncertainty
    confidence = min(combined_score * 100, 75)

    indicators = []
    indicators.extend(perplexity_result.get("indicators", []))
    indicators.extend(burstiness_result.get("indicators", []))
    indicators.extend(vocabulary_result.get("indicators", []))
    indicators.extend(pattern_result.get("indicators", []))

    return {
        "is_ai_generated": is_ai,
        "confidence": round(confidence, 2),
        "method": "statistical_analysis",
        "disclaimer": _get_disclaimer(),
        "details": {
            "perplexity_score": round(perplexity_result["score"] * 100, 2),
            "burstiness_score": round(burstiness_result["score"] * 100, 2),
            "vocabulary_score": round(vocabulary_result["score"] * 100, 2),
            "pattern_score": round(pattern_result["score"] * 100, 2),
            "indicators": indicators,
            "text_stats": {
                "character_count": len(text),
                "word_count": len(text.split()),
                "sentence_count": len(re.split(r'[.!?]+', text)),
            }
        }
    }


def _get_disclaimer() -> str:
    return (
        "WARNING: AI text detection is unreliable. False positives are common, "
        "especially for non-native English speakers, formal writing, and edited text. "
        "Do not use for academic, legal, or employment decisions."
    )


def _analyze_perplexity_proxy(text: str) -> dict:
    """
    Estimate perplexity using character and word-level statistics.

    AI text tends to have lower perplexity (more predictable).
    Note: True perplexity requires a language model - this is a proxy.
    """
    indicators = []
    score = 0.0

    words = text.lower().split()
    if len(words) < 20:
        return {"score": 0.5, "indicators": ["text_too_short"]}

    # 1. Character-level entropy
    char_counts = Counter(text.lower())
    total_chars = sum(char_counts.values())
    char_entropy = -sum(
        (count/total_chars) * math.log2(count/total_chars)
        for count in char_counts.values()
    )

    # Low entropy suggests predictable text (AI-like)
    if char_entropy < 4.0:
        indicators.append("low_character_entropy")
        score += 0.35
    elif char_entropy < 4.5:
        score += 0.15

    # 2. Word-level entropy
    word_counts = Counter(words)
    total_words = len(words)
    word_entropy = -sum(
        (count/total_words) * math.log2(count/total_words)
        for count in word_counts.values()
    )

    # Normalize by log of vocabulary size
    vocab_size = len(word_counts)
    max_entropy = math.log2(vocab_size) if vocab_size > 1 else 1
    normalized_entropy = word_entropy / max_entropy

    if normalized_entropy < 0.7:
        indicators.append("repetitive_word_usage")
        score += 0.3
    elif normalized_entropy < 0.8:
        score += 0.15

    # 3. Bigram predictability
    bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
    bigram_counts = Counter(bigrams)

    # High bigram repetition suggests AI
    unique_bigrams = len(bigram_counts)
    bigram_ratio = unique_bigrams / len(bigrams) if bigrams else 1

    if bigram_ratio < 0.7:
        indicators.append("predictable_word_pairs")
        score += 0.25

    return {"score": min(score, 1.0), "indicators": indicators}


def _analyze_burstiness(text: str) -> dict:
    """
    Analyze sentence-level variation (burstiness).

    Human writing tends to have more variation in sentence length
    and complexity. AI writing is often more uniform.
    """
    indicators = []
    score = 0.0

    # Split into sentences
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

    if len(sentences) < 3:
        return {"score": 0.5, "indicators": ["insufficient_sentences"]}

    # 1. Sentence length variation
    lengths = [len(s.split()) for s in sentences]
    mean_length = sum(lengths) / len(lengths)
    std_length = math.sqrt(sum((l - mean_length)**2 for l in lengths) / len(lengths))

    # Coefficient of variation
    cv = std_length / mean_length if mean_length > 0 else 0

    # Low CV means uniform sentence lengths (AI-like)
    if cv < 0.3:
        indicators.append("uniform_sentence_length")
        score += 0.4
    elif cv < 0.4:
        score += 0.2

    # 2. Sentence start variation
    starts = [s.split()[0].lower() if s.split() else "" for s in sentences]
    unique_starts = len(set(starts))
    start_variety = unique_starts / len(starts)

    if start_variety < 0.5:
        indicators.append("repetitive_sentence_starts")
        score += 0.3

    # 3. Paragraph structure analysis
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if len(paragraphs) > 1:
        para_lengths = [len(p.split()) for p in paragraphs]
        para_cv = (
            math.sqrt(sum((l - sum(para_lengths)/len(para_lengths))**2
                     for l in para_lengths) / len(para_lengths))
            / (sum(para_lengths)/len(para_lengths))
            if sum(para_lengths) > 0 else 0
        )

        if para_cv < 0.25:
            indicators.append("uniform_paragraph_length")
            score += 0.25

    return {"score": min(score, 1.0), "indicators": indicators}


def _analyze_vocabulary(text: str) -> dict:
    """
    Analyze vocabulary patterns.

    AI text often has:
    - Certain overused words/phrases
    - Specific hedging language
    - Formal connectives
    """
    indicators = []
    score = 0.0

    text_lower = text.lower()
    words = text_lower.split()
    word_count = len(words)

    if word_count < 50:
        return {"score": 0.5, "indicators": ["text_too_short"]}

    # 1. AI-associated phrases (common in LLM outputs)
    ai_phrases = [
        "it's important to note",
        "it is important to note",
        "it's worth noting",
        "it is worth noting",
        "in conclusion",
        "in summary",
        "to summarize",
        "first and foremost",
        "last but not least",
        "at the end of the day",
        "it goes without saying",
        "needless to say",
        "as previously mentioned",
        "as mentioned earlier",
        "on the other hand",
        "that being said",
        "having said that",
        "with that in mind",
        "it can be argued",
        "one could argue",
        "it should be noted",
        "moreover",
        "furthermore",
        "additionally",
        "consequently",
        "nevertheless",
        "nonetheless",
        "subsequently",
        "delve",
        "utilize",
        "leverage",
        "facilitate",
        "comprehensive",
        "robust",
        "streamline",
        "synergy",
        "paradigm",
    ]

    phrase_count = sum(1 for phrase in ai_phrases if phrase in text_lower)

    if phrase_count >= 5:
        indicators.append("high_ai_phrase_density")
        score += 0.45
    elif phrase_count >= 3:
        indicators.append("moderate_ai_phrase_usage")
        score += 0.25
    elif phrase_count >= 1:
        score += 0.1

    # 2. Hedging language density
    hedging_words = [
        "may", "might", "could", "would", "should",
        "perhaps", "possibly", "potentially", "likely",
        "generally", "typically", "usually", "often",
        "somewhat", "relatively", "fairly", "quite"
    ]

    hedging_count = sum(words.count(w) for w in hedging_words)
    hedging_ratio = hedging_count / word_count

    if hedging_ratio > 0.05:
        indicators.append("excessive_hedging")
        score += 0.25
    elif hedging_ratio > 0.03:
        score += 0.1

    # 3. Lexical diversity (Type-Token Ratio)
    unique_words = len(set(words))
    ttr = unique_words / word_count

    # Very high TTR can indicate AI (avoids repetition unnaturally)
    if ttr > 0.75 and word_count > 100:
        indicators.append("unusually_high_vocabulary_diversity")
        score += 0.2

    return {"score": min(score, 1.0), "indicators": indicators}


def _analyze_patterns(text: str) -> dict:
    """
    Analyze structural patterns.

    AI text often has:
    - List-heavy structure
    - Consistent formatting
    - Specific transition patterns
    """
    indicators = []
    score = 0.0

    lines = text.split('\n')
    text_lower = text.lower()

    # 1. List detection
    list_patterns = [
        r'^\s*[-•*]\s',  # Bullet points
        r'^\s*\d+[.)]\s',  # Numbered lists
        r'^\s*[a-z][.)]\s',  # Letter lists
    ]

    list_lines = sum(
        1 for line in lines
        if any(re.match(p, line) for p in list_patterns)
    )

    if list_lines > len(lines) * 0.3:
        indicators.append("heavy_list_usage")
        score += 0.2

    # 2. Section header pattern
    header_pattern = r'^#+\s|^[A-Z][A-Za-z\s]+:$'
    headers = sum(1 for line in lines if re.match(header_pattern, line.strip()))

    if headers > 3 and headers > len(lines) * 0.1:
        indicators.append("structured_headers")
        score += 0.15

    # 3. Transition word pattern at sentence starts
    sentences = re.split(r'[.!?]\s+', text)
    transition_starts = [
        "however", "therefore", "furthermore", "moreover",
        "additionally", "consequently", "nevertheless",
        "in addition", "as a result", "on the other hand",
        "first", "second", "third", "finally", "lastly"
    ]

    transition_count = sum(
        1 for s in sentences
        if any(s.lower().strip().startswith(t) for t in transition_starts)
    )

    if len(sentences) > 5 and transition_count > len(sentences) * 0.25:
        indicators.append("excessive_transition_words")
        score += 0.25

    # 4. Quotation and example patterns
    quote_patterns = [
        r'"[^"]{20,}"',  # Quoted text
        r'for example',
        r'for instance',
        r'such as',
    ]

    quote_count = sum(len(re.findall(p, text_lower)) for p in quote_patterns)

    if quote_count > 5:
        indicators.append("heavy_example_usage")
        score += 0.15

    return {"score": min(score, 1.0), "indicators": indicators}
