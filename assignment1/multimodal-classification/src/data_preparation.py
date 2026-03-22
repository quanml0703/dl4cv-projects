"""Data preparation utilities for the CrisisMMD humanitarian dataset.

Responsibilities:
- Merge 8 original labels into 5 final classes (per OVERVIEW specification).
- Create a stratified 200-sample evaluation set from the dev split,
  ensuring every class is represented proportionally.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict

# ─────────────────────────────────────────────────────────
# Label constants
# ─────────────────────────────────────────────────────────

#: Mapping from the 8 original CrisisMMD humanitarian labels to 5 merged labels.
LABEL_MERGE_MAP: dict[str, str] = {
    # keep as-is
    "affected_individuals":                   "affected_individuals",
    "not_humanitarian":                       "not_humanitarian",
    "other_relevant_information":             "other_relevant_information",
    "rescue_volunteering_or_donation_effort": "rescue_volunteering_or_donation_effort",
    # merge into affected_individuals
    "injured_or_dead_people":                 "affected_individuals",
    "missing_or_found_people":                "affected_individuals",
    # merge into infrastructure_and_utility_damage
    "infrastructure_and_utility_damage":      "infrastructure_and_utility_damage",
    "vehicle_damage":                         "infrastructure_and_utility_damage",
}

#: Sorted list of the 5 final class labels.
FINAL_LABELS: list[str] = sorted(set(LABEL_MERGE_MAP.values()))


# ─────────────────────────────────────────────────────────
# Label merging
# ─────────────────────────────────────────────────────────

def _build_label_mapper(label_names: Optional[list[str]] = None):
    """Return a map-function that converts a raw label to a merged string label.

    Args:
        label_names: List of class names when the dataset uses ``ClassLabel``
                     (integer) encoding.  Pass ``None`` for string labels.

    Returns:
        Callable suitable for ``Dataset.map()``.
    """
    def _mapper(example: dict) -> dict:
        raw = example["label"]
        # Decode integer ClassLabel to string if necessary
        if isinstance(raw, int) and label_names is not None:
            raw = label_names[raw]
        example["label"] = LABEL_MERGE_MAP.get(str(raw), str(raw))
        return example

    return _mapper


def apply_label_merging(
    dataset: DatasetDict,
    label_names: Optional[list[str]] = None,
) -> DatasetDict:
    """Apply label merging (8 → 5 classes) to every split in the dataset.

    Args:
        dataset: HuggingFace ``DatasetDict`` with train / dev / test splits.
        label_names: Original class-name list when the ``label`` feature is a
                     ``ClassLabel`` (integer). Obtain via
                     ``dataset["train"].features["label"].names``.

    Returns:
        ``DatasetDict`` where ``label`` is a plain string from ``FINAL_LABELS``.
    """
    from datasets import Value

    mapper = _build_label_mapper(label_names)
    # Cast the label feature to plain string so HuggingFace does not
    # re-encode the mapped values back to ClassLabel integers.
    new_features = dataset["train"].features.copy()
    new_features["label"] = Value("string")
    return dataset.map(mapper, features=new_features)


# ─────────────────────────────────────────────────────────
# Stratified evaluation set
# ─────────────────────────────────────────────────────────

def create_stratified_eval_set(
    dataset: Dataset,
    n_samples: int = 200,
    label_col: str = "label",
    seed: int = 42,
    save_path: Optional[str] = None,
) -> Dataset:
    """Create a stratified evaluation subset ensuring all classes are covered.

    Samples are allocated proportionally across classes (minimum 1 per class).
    Any rounding discrepancy is absorbed by the most frequent class so that
    the total is exactly ``n_samples``.

    Args:
        dataset: A single HuggingFace ``Dataset`` split (e.g. the dev split)
                 **after** label merging has been applied.
        n_samples: Total number of samples to select (default: 200).
        label_col: Column name holding the (string) label.
        seed: Random seed for reproducibility.
        save_path: If provided, save the metadata (without PIL images) as CSV.

    Returns:
        A HuggingFace ``Dataset`` of length ``n_samples`` (or less if the
        split is smaller than requested).
    """
    np.random.seed(seed)

    labels: list[str] = dataset[label_col]

    # Build class → sample-index mapping
    class_indices: dict[str, list[int]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        class_indices[lbl].append(i)

    total = len(labels)
    class_counts = {lbl: len(idxs) for lbl, idxs in class_indices.items()}

    # --- Proportional allocation with minimum guarantee ---
    samples_per_class: dict[str, int] = {}
    for lbl, count in class_counts.items():
        allocated = max(1, int(round(count / total * n_samples)))
        samples_per_class[lbl] = allocated

    # Correct rounding drift so the sum equals exactly n_samples
    diff = n_samples - sum(samples_per_class.values())
    largest_class = max(class_counts, key=class_counts.get)
    samples_per_class[largest_class] = max(
        1, samples_per_class[largest_class] + diff
    )

    # --- Stratified sampling ---
    selected_indices: list[int] = []
    for lbl, n in samples_per_class.items():
        indices = class_indices[lbl]
        n = min(n, len(indices))
        chosen = np.random.choice(indices, size=n, replace=False).tolist()
        selected_indices.extend(chosen)

    eval_ds = dataset.select(selected_indices)

    # Optionally persist metadata (skip non-serialisable image column)
    if save_path:
        meta_cols = [c for c in eval_ds.column_names if c != "image"]
        meta_df = pd.DataFrame({c: eval_ds[c] for c in meta_cols})
        meta_df.to_csv(save_path, index=False)
        print(f"[data_preparation] Eval set metadata saved → {save_path}")

    return eval_ds


def summarise_eval_set(eval_ds: Dataset, label_col: str = "label") -> pd.DataFrame:
    """Return a DataFrame summarising the class distribution of the eval set.

    Args:
        eval_ds: The stratified evaluation ``Dataset``.
        label_col: Column name for labels.

    Returns:
        DataFrame with columns ``['label', 'count', 'proportion']``.
    """
    labels = eval_ds[label_col]
    counts = pd.Series(labels).value_counts().reset_index()
    counts.columns = ["label", "count"]
    counts["proportion"] = (counts["count"] / counts["count"].sum()).round(3)
    return counts.sort_values("label").reset_index(drop=True)
