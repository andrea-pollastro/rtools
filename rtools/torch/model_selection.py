from typing import Union, Tuple
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split as tts
import numpy as np

def train_test_split(
    dataset: Union[Dataset, Subset],
    *args,
    **kwargs
) -> Tuple[Subset, Subset]:
    """
    Train/test split for PyTorch Dataset or Subset.

    Contract:
    - Accepts Dataset
    - Accepts Subset whose parent is a Dataset
    - Raises on nested Subset input
    """
    if not isinstance(dataset, (Dataset, Subset)):
        raise TypeError(
            f"Expected torch Dataset or Subset, got {type(dataset)}"
        )

    if isinstance(dataset, Subset):
        if isinstance(dataset.dataset, Subset):
            raise ValueError(
                "Nested Subset detected. "
                "train_test_split expects a Dataset or a Subset of a Dataset."
            )
        base_dataset = dataset.dataset
        indices = np.asarray(dataset.indices)
    else:
        base_dataset = dataset
        indices = np.arange(len(dataset))  # type: ignore

    train_idx, test_idx = tts(indices, *args, **kwargs)

    return (
        Subset(base_dataset, train_idx.tolist()),
        Subset(base_dataset, test_idx.tolist()),
    )