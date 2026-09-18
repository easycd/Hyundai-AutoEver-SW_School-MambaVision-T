"""Dataset inventory and reproducible split; no ML dependencies required."""
import random
from pathlib import Path

CLASSES = {"정상": 0, "반칙": 1}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp"}


def collect(root, split):
    root = Path(root)
    rows = []
    for name, label in CLASSES.items():
        folder = root / name / split
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTENSIONS)
        if not files:
            raise ValueError(f"No images: {folder}")
        rows.extend({"path": p.relative_to(root).as_posix(), "label": label} for p in files)
    return rows


def split_train(rows, fraction=0.2, seed=42):
    if not 0 < fraction < 1:
        raise ValueError("val-fraction must be between 0 and 1")
    rng = random.Random(seed)
    train, val = [], []
    for label in CLASSES.values():
        group = [dict(row) for row in rows if row["label"] == label]
        if len(group) < 2:
            raise ValueError("Each class needs at least two training images")
        rng.shuffle(group)
        n = max(1, min(len(group) - 1, round(len(group) * fraction)))
        val.extend(group[:n])
        train.extend(group[n:])
    return train, val


def metrics(targets, predictions):
    cm = [[0, 0], [0, 0]]
    for target, prediction in zip(targets, predictions):
        cm[target][prediction] += 1
    tn, fp = cm[0]
    fn, tp = cm[1]
    def divide(a, b):
        return a / b if b else 0.0
    return {
        "accuracy": divide(tn + tp, len(targets)),
        "violation_precision": divide(tp, tp + fp),
        "violation_recall": divide(tp, tp + fn),
        "violation_f1": divide(2 * tp, 2 * tp + fp + fn),
        "normal_false_positive_rate": divide(fp, tn + fp),
        "confusion_matrix": cm,
    }
