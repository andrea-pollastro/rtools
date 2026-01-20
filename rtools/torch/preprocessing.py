from torch.utils.data import Dataset
import torch

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
