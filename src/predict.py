"""Prediction interface for the document classifier.

Single entry point for classifying a document against a trained model
bundle. All threshold and fallback routing lives here so the API layer
and the evaluation code share identical behavior.
"""

from typing import Any

DEFAULT_THRESHOLDS: dict[str, float] = {
    "auto_accept": 0.90,
    "manual_review": 0.70,
    "other": 0.55,
}

FALLBACK_LABEL = "other"


def predict_text(text: str, bundle: dict[str, Any], top_k: int = 3) -> dict[str, Any]:
    """Classify ``text`` using a model bundle and route it by confidence.

    Args:
        text: Raw document text to classify.
        bundle: Model bundle dict. Must contain a
            ``model`` (sklearn pipeline supporting ``predict_proba``) and
            ``target_names`` index-aligned with the probability output.
            ``confidence_thresholds`` is optional; missing keys fall back
            to defaults of 0.90 / 0.70 / 0.55.
        top_k: Number of (label, confidence) pairs to return, clamped to
            the number of available classes.

    Returns:
        A dict with:
            label: Final label after threshold routing (may be "other").
            raw_label: The model's argmax label, before routing.
            confidence: Maximum predicted probability.
            decision: One of "auto_accept", "review_recommended",
                "manual_review", or "fallback_other".
            top_k: List of ``{"label": str, "confidence": float}``
                sorted by confidence descending.
    """
    model = bundle["model"]
    target_names: list[str] = list(bundle["target_names"])

    thresholds = {**DEFAULT_THRESHOLDS, **bundle.get("confidence_thresholds", {})}

    probabilities = model.predict_proba([text])[0]

    ranked = sorted(
        zip(target_names, probabilities, strict=True),
        key=lambda pair: pair[1],
        reverse=True,
    )

    raw_label, confidence = ranked[0]
    confidence = float(confidence)

    if confidence >= thresholds["auto_accept"]:
        label, decision = raw_label, "auto_accept"
    elif confidence >= thresholds["manual_review"]:
        label, decision = raw_label, "review_recommended"
    elif confidence < thresholds["other"]:
        label, decision = FALLBACK_LABEL, "fallback_other"
    else:
        label, decision = raw_label, "manual_review"

    k = max(1, min(top_k, len(ranked)))
    top = [{"label": name, "confidence": float(prob)} for name, prob in ranked[:k]]

    return {
        "label": label,
        "raw_label": raw_label,
        "confidence": confidence,
        "decision": decision,
        "top_k": top,
    }
