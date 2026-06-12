"""Dataset loading with label aliasing and exact-duplicate removal.

Reads category folders of ``.txt`` files, normalizes folder names
through the alias map (``technologie`` -> ``technology``), keeps the
``other`` folder aside as an out-of-distribution holdout, and removes
exact duplicates (hash of whitespace/case-normalized text) across the
whole known-label pool *before* any train/validation/test split so a
duplicated document can never leak across splits.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from src.config import EXCLUDED_TRAINING_LABELS, LABEL_ALIASES, TRAINED_LABELS


@dataclass(frozen=True)
class DatasetBundle:
    """Everything downstream code needs, loaded once and immutable.

    Attributes:
        known_texts: Deduplicated document texts for the trained labels.
        known_labels: Canonical label per text (aligned with known_texts).
        known_paths: Source file path per text (aligned with known_texts).
        target_names: Sorted, stable list of trained labels. Matches the
            column order of the saved model's predict_proba output.
        other_texts: The out-of-distribution holdout documents.
        other_paths: Source file paths for the holdout documents.
        duplicate_report: Removed-duplicate counts per label, plus
            "total_removed" and "cross_class_removed" summary keys.
        empty_files: Paths of files skipped because they contain no text.
    """

    known_texts: list[str]
    known_labels: list[str]
    known_paths: list[str]
    target_names: list[str]
    other_texts: list[str]
    other_paths: list[str]
    duplicate_report: dict[str, int]
    empty_files: list[str] = field(default_factory=list)


def normalize_for_hashing(text: str) -> str:
    """Lowercase and collapse all whitespace so trivially reformatted
    copies of the same document hash identically."""
    return " ".join(text.lower().split())


def text_fingerprint(text: str) -> str:
    """Stable SHA-256 fingerprint of the normalized text."""
    return hashlib.sha256(normalize_for_hashing(text).encode("utf-8")).hexdigest()


def _read_text_files(folder: Path) -> list[tuple[str, str]]:
    """Read every .txt file in a folder as (path, text), sorted by name."""
    return [
        (str(path), path.read_text(encoding="utf-8", errors="ignore"))
        for path in sorted(folder.glob("*.txt"))
    ]


def load_dataset(data_dir: str | Path) -> DatasetBundle:
    """Load the dataset from ``data_dir`` into a :class:`DatasetBundle`.

    Folder names are validated rather than trusted: anything that is not
    a trained label, a known alias, or an excluded holdout folder raises
    a ``ValueError`` so silent label drift is impossible.

    Duplicate removal keeps the first occurrence in deterministic
    (label, filename) order and applies to the known pool only; the
    ``other`` holdout files are loaded verbatim.
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    folders = sorted(p for p in root.iterdir() if p.is_dir())
    if not folders:
        raise ValueError(f"No category folders found under {root}")

    unexpected = [
        f.name
        for f in folders
        if LABEL_ALIASES.get(f.name, f.name) not in (*TRAINED_LABELS, *EXCLUDED_TRAINING_LABELS)
    ]
    if unexpected:
        raise ValueError(
            f"Unexpected category folders {unexpected}; "
            "add them to the alias map or label list before training."
        )

    known_texts: list[str] = []
    known_labels: list[str] = []
    known_paths: list[str] = []
    other_texts: list[str] = []
    other_paths: list[str] = []
    empty_files: list[str] = []

    seen_hashes: dict[str, str] = {}  # fingerprint -> first label seen
    duplicate_report: dict[str, int] = {"total_removed": 0, "cross_class_removed": 0}

    for folder in folders:
        label = LABEL_ALIASES.get(folder.name, folder.name)
        for path, text in _read_text_files(folder):
            if not text.strip():
                empty_files.append(path)
                continue
            if label in EXCLUDED_TRAINING_LABELS:
                other_texts.append(text)
                other_paths.append(path)
                continue
            fingerprint = text_fingerprint(text)
            if fingerprint in seen_hashes:
                duplicate_report[label] = duplicate_report.get(label, 0) + 1
                duplicate_report["total_removed"] += 1
                if seen_hashes[fingerprint] != label:
                    duplicate_report["cross_class_removed"] += 1
                continue
            seen_hashes[fingerprint] = label
            known_texts.append(text)
            known_labels.append(label)
            known_paths.append(path)

    target_names = sorted(set(known_labels))
    if target_names != TRAINED_LABELS:
        raise ValueError(f"Loaded labels {target_names} do not match the expected trained labels.")

    return DatasetBundle(
        known_texts=known_texts,
        known_labels=known_labels,
        known_paths=known_paths,
        target_names=target_names,
        other_texts=other_texts,
        other_paths=other_paths,
        duplicate_report=duplicate_report,
        empty_files=empty_files,
    )
