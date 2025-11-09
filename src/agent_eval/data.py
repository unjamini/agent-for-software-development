from __future__ import annotations

from typing import Iterable, List, Mapping

from datasets import load_dataset


def load_humanevalfix(
    *,
    subset: str = "python",
    dataset_id_candidates: Iterable[str] | None = None,
) -> List[Mapping[str, object]]:
    """Load the HumanEvalFix benchmark split.

    Parameters
    ----------
    subset : str
        Dataset subset (e.g., "python")
    dataset_id_candidates : Iterable[str] | None
        List of Hugging Face dataset IDs to try

    Returns
    -------
    List[Mapping[str, object]]
        List of dataset examples
    """

    try:
        dataset = load_dataset("bigcode/humanevalpack", subset, split="test")
        return list(dataset)
    except Exception as e:
        raise ValueError(f"Could not load bigcode/humanevalpack ({subset}). Error: {e}") from e

    if dataset_id_candidates is None:
        dataset_id_candidates = (
            "bigcode/humanevalfix",
            "bigcode/humanevalfix-python",
        )

    err: Exception | None = None
    for ds_id in dataset_id_candidates:
        try:
            dataset = load_dataset(ds_id, subset, split="test")
        except Exception as exc:
            err = exc
            continue
        return list(dataset)

    raise ValueError("Could not fetch HumanEvalFix") from err


