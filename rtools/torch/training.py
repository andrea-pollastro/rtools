import os
import torch
import platform
from typing import (
    Union, 
    List, 
    Tuple,
    Dict,
)

def to_device(batch: Union[List, Tuple, torch.Tensor], device: str) -> Union[Tuple, torch.Tensor]:
    """
    Move a batch or tensor to the specified device.

    Parameters
    ----------
    batch : Union[List, Tuple, torch.Tensor]
        The batch to move. Can be a tensor, or a list/tuple of tensors.
    device : str
        The target device (e.g., 'cpu', 'cuda', 'cuda:0').

    Returns
    -------
    Union[Tuple, torch.Tensor]
        The batch on the target device. If input is a list/tuple, returns a tuple.
    """
    if isinstance(batch, (tuple, list)):
        return tuple(b.to(device) for b in batch)
    else:
        return batch.to(device)
    
class RunningStats:
    """
    Tracker for running statistics of multiple metrics.

    Maintains cumulative totals and counts for a set of registered metrics,
    allowing efficient computation of running means without storing all
    individual values.

    Parameters
    ----------
    metrics : List[str]
        List of metric names to track.

    Notes
    -----
    The running mean is computed as the weighted average: mean = total / count.
    """
    def __init__(self, metrics: List[str]):
        self.metrics = metrics
        self.reset()

    def reset(self) -> None:
        """Reset all metric totals and count to zero."""
        self.total: Dict[str, float] = {m: 0.0 for m in self.metrics}
        self.count = 0

    def update(self, values: Dict[str, float], elem: int) -> None:
        """
        Update running totals with new metric values.

        Parameters
        ----------
        values : Dict[str, float]
            Dictionary mapping metric names to their new values.
        elem : int, optional
            Weight for this update (default=1). Allows batch updates.

        Raises
        ------
        KeyError
            If a metric name in `values` was not registered in `__init__`.
        """
        for m, v in values.items():
            if m not in self.total:
                raise KeyError(f"Metric '{m}' was not registered.")
            self.total[m] += v * elem
        self.count += elem

    def mean(self) -> Dict[str, float]:
        """
        Compute the running mean for all tracked metrics.

        Returns
        -------
        Dict[str, float]
            Dictionary mapping metric names to their current running means.

        Raises
        ------
        RuntimeError
            If called before any update has been made.
        """
        if self.count == 0:
            raise RuntimeError("RunningStats.mean() called before any update.")

        return {
            m: self.total[m] / self.count
            for m in self.metrics
        }
