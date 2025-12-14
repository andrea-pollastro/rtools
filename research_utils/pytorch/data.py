from sklearn.model_selection import train_test_split as tts
from torch.utils.data import Dataset, Subset
from typing import Union, Tuple, Optional
import numpy as np
import torch


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


class StandardizedDataset(Dataset):
    """
    Dataset wrapper that applies feature-wise standardization.

    This class mimics the behavior of sklearn.preprocessing.StandardScaler
    in a PyTorch-native way. It applies the transformation:

        x ↦ (x - mean) / std

    lazily at data access time, without modifying or copying the underlying data.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The underlying dataset. Each item must return a tuple whose first
        element is a tensor of features.
    mean : torch.Tensor
        Feature-wise mean, typically computed from the training dataset.
        Shape must be broadcastable to the feature tensor.
    std : torch.Tensor
        Feature-wise standard deviation, typically computed from the training
        dataset. Shape must be broadcastable to the feature tensor.

    Notes
    -----
    - The transformation is applied lazily in `__getitem__`, so memory usage
      is minimal and compatible with DataLoader, shuffling, and multiprocessing.
    """
    def __init__(self, dataset: Dataset, mean: torch.Tensor, std: torch.Tensor):
        self.dataset = dataset
        self.mean = mean
        self.std = torch.where(std == 0, torch.ones_like(std), std) # to guard against zeros


    def __len__(self):
        return len(self.dataset) # type: ignore

    def __getitem__(self, idx):
        x, *rest = self.dataset[idx]
        x = (x - self.mean) / self.std
        return (x, *rest)


import torch
from torch.utils.data import Dataset, DataLoader


def compute_mean_std(
    dataset: Dataset,
    batch_size: int = 1024,
    num_workers: int = 0,
):
    """
    Compute feature-wise mean and standard deviation of a PyTorch Dataset.

    This function computes dataset-level statistics in a streaming fashion,
    without loading the entire dataset into memory. It is equivalent to
    computing:

        mean = X.mean(dim=0)
        std  = X.std(dim=0, unbiased=False)

    where X is the concatenation of all samples in the dataset, but is safe
    for large datasets that do not fit in memory.

    The computation uses a numerically stable, order-independent algorithm
    (a batch-wise form of Welford / Chan variance update), and produces exact
    results up to floating-point precision.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset yielding samples of the form (x, ...) where `x` is a tensor
        of features. Only the first element of each sample is used.
    batch_size : int, default=1024
        Number of samples per batch used to compute intermediate statistics.
        Does not affect correctness, only memory usage and performance.
    num_workers : int, default=0
        Number of worker processes used by the DataLoader.

    Returns
    -------
    mean : torch.Tensor
        Feature-wise mean of the dataset. Shape matches the feature tensor
        excluding the batch dimension.
    std : torch.Tensor
        Feature-wise standard deviation of the dataset, computed as the
        population standard deviation (unbiased=False), matching the
        behavior of sklearn.preprocessing.StandardScaler.

    Notes
    -----
    - The variance is computed as population variance (division by N),
      not sample variance (division by N - 1).
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    n = 0
    mean: Optional[torch.Tensor] = None
    var: Optional[torch.Tensor] = None

    for x, *_ in loader:
        x = x.float()
        batch_n = x.size(0)

        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)

        if mean is None:
            mean = batch_mean
            var = batch_var
            n = batch_n
            continue

        assert var is not None

        delta = batch_mean - mean
        tot_n = n + batch_n

        mean = mean + delta * batch_n / tot_n
        var = (
            n * var
            + batch_n * batch_var
            + delta.pow(2) * n * batch_n / tot_n
        ) / tot_n
        n = tot_n

    if mean is None or var is None:
        raise ValueError("Dataset must not be empty")

    std = torch.sqrt(var)
    return mean, std
