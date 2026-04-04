"""Utility functions for PyTorch operations."""

from typing import Any, Union
import torch


def to_device(data: Any, device: Union[str, torch.device]) -> Any:
    """
    Move tensors to a specified device.

    Recursively handles tensors in tuples, lists, dictionaries, and other structures.
    Non-tensor objects are returned unchanged.

    Parameters
    ----------
    data : Any
        Data structure containing tensors (tensor, tuple, list, dict, etc.).
    device : str or torch.device
        Target device (e.g., 'cpu', 'cuda', 'cuda:0').

    Returns
    -------
    Any
        Data structure with all tensors moved to the target device.

    Examples
    --------
    >>> import torch
    >>> from rtools.torch.utils import to_device
    >>> x = torch.randn(3, 4)
    >>> y = to_device(x, 'cpu')
    >>> to_device((x, y), 'cpu')
    (tensor(...), tensor(...))
    >>> to_device({'a': x, 'b': y}, 'cpu')
    {'a': tensor(...), 'b': tensor(...)}
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        result = [to_device(item, device) for item in data]
        return type(data)(result)
    else:
        return data
