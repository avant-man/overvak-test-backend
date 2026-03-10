"""
Metrics module: computes per-class TP/FP/FN and macro Precision/Recall/F1
for the 5 labeled evaluation videos.
"""

CLASSES = ["Climbing", "Crying", "Falling", "Hitting", "Jumping", "Laughing", "Pushing", "Running"]

GROUND_TRUTH: dict[str, dict[str, int]] = {
    "VkhiJBpoofM": {
        "Climbing": 0, "Crying": 0, "Falling": 0, "Hitting": 0,
        "Jumping": 0, "Laughing": 1, "Pushing": 0, "Running": 1,
    },
    "qzvfds5Nqus": {
        "Climbing": 0, "Crying": 1, "Falling": 1, "Hitting": 1,
        "Jumping": 0, "Laughing": 1, "Pushing": 0, "Running": 0,
    },
    "D0x6P-UhGgs": {
        "Climbing": 0, "Crying": 1, "Falling": 0, "Hitting": 1,
        "Jumping": 0, "Laughing": 0, "Pushing": 0, "Running": 0,
    },
    "v8RPGCc8p9o": {
        "Climbing": 1, "Crying": 1, "Falling": 1, "Hitting": 1,
        "Jumping": 0, "Laughing": 1, "Pushing": 0, "Running": 0,
    },
    "aGw3sH8aVo8": {
        "Climbing": 0, "Crying": 1, "Falling": 0, "Hitting": 0,
        "Jumping": 0, "Laughing": 0, "Pushing": 1, "Running": 0,
    },
}


def compute_metrics(video_id: str, predictions: dict[str, int]) -> dict | None:
    """
    Compute per-class and macro metrics for a labeled video.

    Args:
        video_id: YouTube video ID to look up in GROUND_TRUTH.
        predictions: Binary prediction dict (Title-case keys, 0 or 1 values).

    Returns:
        A dict with per-class results and macro Precision, Recall, F1,
        or None if video_id is not in GROUND_TRUTH.
    """
    gt = GROUND_TRUTH.get(video_id)
    if gt is None:
        return None

    per_class: dict[str, dict[str, int]] = {}
    total_tp = total_fp = total_fn = 0

    for cls in CLASSES:
        pred = predictions.get(cls, 0)
        truth = gt[cls]
        tp = int(pred == 1 and truth == 1)
        fp = int(pred == 1 and truth == 0)
        fn = int(pred == 0 and truth == 1)
        tn = int(pred == 0 and truth == 0)
        per_class[cls] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "per_class":  per_class,
        "tp":         total_tp,
        "fp":         total_fp,
        "fn":         total_fn,
        "precision":  round(precision, 4),
        "recall":     round(recall, 4),
        "f1":         round(f1, 4),
        "ground_truth": gt,
    }


def format_metrics_table(metrics: dict) -> str:
    """
    Render metrics as a plain-text table.

    Args:
        metrics: Dict returned by compute_metrics().

    Returns:
        A formatted string table with per-class TP/FP/FN/TN and macro scores.
    """
    lines: list[str] = []
    header = f"{'Class':<12} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}"
    lines.append(header)
    lines.append("-" * len(header))

    for cls in CLASSES:
        row = metrics["per_class"][cls]
        lines.append(
            f"{cls:<12} {row['tp']:>4} {row['fp']:>4} {row['fn']:>4} {row['tn']:>4}"
        )

    lines.append("-" * len(header))
    lines.append(
        f"{'Macro':<12} "
        f"P={metrics['precision']:.3f}  "
        f"R={metrics['recall']:.3f}  "
        f"F1={metrics['f1']:.3f}"
    )
    return "\n".join(lines)
