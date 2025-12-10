"""
Evaluation utilities for quality metrics.

Provides functions for calculating:
- WER (Word Error Rate) for STT evaluation
- BLEU score for translation quality
- COMET score for neural translation evaluation
"""

import re
from typing import Optional


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.

    - Lowercase
    - Remove punctuation
    - Normalize whitespace
    """
    text = text.lower()
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Normalize whitespace
    text = " ".join(text.split())
    return text


def tokenize(text: str) -> list[str]:
    """Simple whitespace tokenization."""
    return normalize_text(text).split()


def calculate_wer(
    hypothesis: str,
    reference: str,
    normalize: bool = True,
) -> float:
    """
    Calculate Word Error Rate (WER).

    WER = (S + D + I) / N
    where:
    - S = substitutions
    - D = deletions
    - I = insertions
    - N = words in reference

    Args:
        hypothesis: Transcribed/predicted text
        reference: Ground truth text
        normalize: Whether to normalize texts before comparison

    Returns:
        WER as a float (0.0 = perfect, 1.0+ = many errors)
    """
    if normalize:
        hyp_words = tokenize(hypothesis)
        ref_words = tokenize(reference)
    else:
        hyp_words = hypothesis.split()
        ref_words = reference.split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else float("inf")

    # Dynamic programming for edit distance
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def calculate_bleu(
    hypothesis: str,
    reference: str | list[str],
    max_n: int = 4,
    smooth: bool = True,
) -> float:
    """
    Calculate BLEU score for translation quality.

    Uses sacrebleu for proper BLEU calculation if available,
    otherwise falls back to a simple implementation.

    Args:
        hypothesis: Translated text
        reference: Reference translation(s)
        max_n: Maximum n-gram order (default 4 for BLEU-4)
        smooth: Whether to use smoothing for short sentences

    Returns:
        BLEU score (0-100, higher is better)
    """
    try:
        import sacrebleu

        # sacrebleu expects list of references
        if isinstance(reference, str):
            references = [[reference]]
        else:
            references = [reference]

        bleu = sacrebleu.corpus_bleu([hypothesis], references)
        return bleu.score

    except ImportError:
        # Fallback to simple BLEU implementation
        return _simple_bleu(hypothesis, reference, max_n, smooth)


def _simple_bleu(
    hypothesis: str,
    reference: str | list[str],
    max_n: int = 4,
    smooth: bool = True,
) -> float:
    """Simple BLEU implementation without external dependencies."""
    import math
    from collections import Counter

    if isinstance(reference, list):
        reference = reference[0]  # Use first reference

    hyp_tokens = tokenize(hypothesis)
    ref_tokens = tokenize(reference)

    if len(hyp_tokens) == 0:
        return 0.0

    # Calculate n-gram precisions
    precisions = []

    for n in range(1, max_n + 1):
        hyp_ngrams = Counter(
            tuple(hyp_tokens[i : i + n]) for i in range(len(hyp_tokens) - n + 1)
        )
        ref_ngrams = Counter(
            tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1)
        )

        # Count matching n-grams (clipped)
        matches = sum(
            min(count, ref_ngrams.get(ngram, 0))
            for ngram, count in hyp_ngrams.items()
        )

        total = sum(hyp_ngrams.values())

        if total == 0:
            precision = 0.0
        elif smooth and matches == 0:
            # Add-1 smoothing
            precision = 1.0 / (total + 1)
        else:
            precision = matches / total

        precisions.append(precision)

    # Geometric mean of precisions
    if any(p == 0 for p in precisions):
        geo_mean = 0.0
    else:
        log_sum = sum(math.log(p) for p in precisions if p > 0)
        geo_mean = math.exp(log_sum / len(precisions))

    # Brevity penalty
    if len(hyp_tokens) >= len(ref_tokens):
        bp = 1.0
    else:
        bp = math.exp(1 - len(ref_tokens) / len(hyp_tokens))

    return bp * geo_mean * 100  # Scale to 0-100


def calculate_comet(
    hypothesis: str,
    reference: str,
    source: str,
    model_name: str = "Unbabel/wmt22-comet-da",
) -> float:
    """
    Calculate COMET score for neural translation quality evaluation.

    COMET uses a trained neural model to evaluate translation quality,
    considering source, hypothesis, and reference.

    Args:
        hypothesis: Translated text
        reference: Reference translation
        source: Original source text
        model_name: COMET model to use

    Returns:
        COMET score (0-1, higher is better)

    Note:
        Requires the unbabel-comet package and model download on first use.
    """
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        raise ImportError(
            "unbabel-comet is required for COMET scoring. "
            "Install with: pip install unbabel-comet"
        )

    # Download and load model (cached after first download)
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)

    # Prepare data
    data = [{"src": source, "mt": hypothesis, "ref": reference}]

    # Get predictions
    output = model.predict(data, batch_size=1, gpus=0)

    # Return system-level score
    return output.system_score


def calculate_comet_batch(
    hypotheses: list[str],
    references: list[str],
    sources: list[str],
    model_name: str = "Unbabel/wmt22-comet-da",
) -> dict:
    """
    Calculate COMET scores for multiple translations (more efficient).

    Args:
        hypotheses: List of translated texts
        references: List of reference translations
        sources: List of original source texts
        model_name: COMET model to use

    Returns:
        Dictionary with:
        - 'scores': List of per-sentence scores
        - 'system_score': Overall system score
    """
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        raise ImportError(
            "unbabel-comet is required for COMET scoring. "
            "Install with: pip install unbabel-comet"
        )

    if not (len(hypotheses) == len(references) == len(sources)):
        raise ValueError("All input lists must have the same length")

    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)

    data = [
        {"src": src, "mt": hyp, "ref": ref}
        for src, hyp, ref in zip(sources, hypotheses, references)
    ]

    output = model.predict(data, batch_size=8, gpus=0)

    return {
        "scores": output.scores,
        "system_score": output.system_score,
    }


def calculate_ter(
    hypothesis: str,
    reference: str,
    normalize: bool = True,
) -> float:
    """
    Calculate Translation Edit Rate (TER).

    TER = (edits) / (reference_length)

    Similar to WER but considers translation edits.

    Args:
        hypothesis: Translated text
        reference: Reference translation
        normalize: Whether to normalize texts

    Returns:
        TER as a float (0.0 = perfect, lower is better)
    """
    # TER uses the same calculation as WER but in translation context
    return calculate_wer(hypothesis, reference, normalize)


def calculate_cer(
    hypothesis: str,
    reference: str,
) -> float:
    """
    Calculate Character Error Rate (CER).

    Useful for Chinese text evaluation.

    Args:
        hypothesis: Transcribed/predicted text
        reference: Ground truth text

    Returns:
        CER as a float (0.0 = perfect)
    """
    hyp_chars = list(hypothesis.replace(" ", ""))
    ref_chars = list(reference.replace(" ", ""))

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else float("inf")

    # Dynamic programming for edit distance at character level
    d = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]

    for i in range(len(ref_chars) + 1):
        d[i][0] = i
    for j in range(len(hyp_chars) + 1):
        d[0][j] = j

    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(ref_chars)][len(hyp_chars)] / len(ref_chars)


class QualityEvaluator:
    """
    Convenience class for running multiple quality metrics.

    Caches COMET model to avoid repeated loading.
    """

    def __init__(self, comet_model: Optional[str] = None):
        """
        Initialize evaluator.

        Args:
            comet_model: COMET model name (None to skip COMET)
        """
        self.comet_model = comet_model
        self._comet_loaded = False
        self._comet = None

    def evaluate_stt(
        self,
        hypothesis: str,
        reference: str,
    ) -> dict:
        """
        Evaluate STT quality.

        Returns:
            Dictionary with WER and CER scores
        """
        return {
            "wer": calculate_wer(hypothesis, reference),
            "cer": calculate_cer(hypothesis, reference),
        }

    def evaluate_translation(
        self,
        hypothesis: str,
        reference: str,
        source: Optional[str] = None,
    ) -> dict:
        """
        Evaluate translation quality.

        Returns:
            Dictionary with BLEU and optionally COMET scores
        """
        result = {
            "bleu": calculate_bleu(hypothesis, reference),
            "ter": calculate_ter(hypothesis, reference),
        }

        if self.comet_model and source:
            try:
                result["comet"] = calculate_comet(
                    hypothesis, reference, source, self.comet_model
                )
            except Exception as e:
                result["comet_error"] = str(e)

        return result

    def evaluate_batch(
        self,
        hypotheses: list[str],
        references: list[str],
        sources: Optional[list[str]] = None,
    ) -> dict:
        """
        Evaluate multiple translations.

        Returns:
            Dictionary with aggregated metrics
        """
        n = len(hypotheses)

        bleu_scores = [
            calculate_bleu(h, r) for h, r in zip(hypotheses, references)
        ]
        ter_scores = [
            calculate_ter(h, r) for h, r in zip(hypotheses, references)
        ]

        result = {
            "bleu_mean": sum(bleu_scores) / n,
            "bleu_scores": bleu_scores,
            "ter_mean": sum(ter_scores) / n,
            "ter_scores": ter_scores,
        }

        if self.comet_model and sources:
            try:
                comet_result = calculate_comet_batch(
                    hypotheses, references, sources, self.comet_model
                )
                result["comet_mean"] = comet_result["system_score"]
                result["comet_scores"] = comet_result["scores"]
            except Exception as e:
                result["comet_error"] = str(e)

        return result
