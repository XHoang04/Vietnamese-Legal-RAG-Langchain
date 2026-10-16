"""Generation metrics: refusal accuracy and validity warning accuracy."""


def compute_refusal_accuracy(
    answer: str,
    expected_behavior: str,
    insufficient_context: bool = False,
) -> float:
    """Evaluate whether system appropriately refused or gave up on out-of-scope/unanswerable requests."""
    refusal_keywords = [
        "ngoài phạm vi",
        "không thuộc phạm vi",
        "không tìm thấy",
        "chưa đủ thông tin",
        "không thể trả lời",
        "từ chối",
    ]
    has_refusal_text = any(kw in answer.lower() for kw in refusal_keywords)
    system_refused = has_refusal_text or insufficient_context

    if expected_behavior == "refuse":
        return 1.0 if system_refused else 0.0
    else:
        # Expected to answer
        return 0.0 if (system_refused and not insufficient_context) else 1.0


def compute_validity_accuracy(
    warnings: list[str],
    expected_validity: str,
) -> float:
    """Evaluate whether validity warnings match expected document status."""
    if expected_validity == "none":
        return 1.0

    all_warn_text = " ".join(warnings).lower()

    if expected_validity == "expired":
        return 1.0 if ("hết hiệu lực" in all_warn_text or "bị thay thế" in all_warn_text) else 0.0
    elif expected_validity == "partial":
        return 1.0 if "hết hiệu lực một phần" in all_warn_text else 0.0
    elif expected_validity == "in_force":
        # In-force documents should not have expired warnings
        return 1.0 if "hết hiệu lực toàn bộ" not in all_warn_text else 0.0

    return 1.0
