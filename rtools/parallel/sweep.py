"""Parallel processing utilities for parameter sweeps."""

from multiprocessing import get_context
from typing import Callable, List, Any, Optional, Tuple, Iterable


def parallel_map(
    func: Callable,
    iterable: Iterable[Any],
    n_jobs: Optional[int] = None,
    start_method: str = "spawn",
) -> List[Any]:
    """
    Apply function to items in an iterable using multiprocessing.

    Parameters
    ----------
    func : Callable
        Function to apply to each item.
    iterable : Iterable
        Iterable of items to process.
    n_jobs : int, optional
        Number of worker processes. If None, uses CPU count.
    start_method : str, default="spawn"
        Process start method: "spawn", "fork", or "forkserver".

    Returns
    -------
    List
        List of results in the same order as input iterable.

    Examples
    --------
    >>> def square(x):
    ...     return x ** 2
    >>> results = parallel_map(square, [1, 2, 3, 4], n_jobs=2)
    >>> results
    [1, 4, 9, 16]
    """
    ctx = get_context(start_method)
    with ctx.Pool(n_jobs) as pool:
        results = pool.map(func, iterable)
    return results


def parallel_starmap(
    func: Callable,
    iterable: Iterable[Tuple[Any, ...]],
    n_jobs: Optional[int] = None,
    start_method: str = "spawn",
) -> List[Any]:
    """
    Apply function to unpacked argument tuples using multiprocessing.

    Parameters
    ----------
    func : Callable
        Function to apply. Should accept unpacked arguments from tuples.
    iterable : Iterable[Tuple]
        Iterable of argument tuples to unpack and process.
    n_jobs : int, optional
        Number of worker processes. If None, uses CPU count.
    start_method : str, default="spawn"
        Process start method: "spawn", "fork", or "forkserver".

    Returns
    -------
    List
        List of results in the same order as input iterable.

    Examples
    --------
    >>> def multiply(x, y):
    ...     return x * y
    >>> results = parallel_starmap(multiply, [(2, 3), (4, 5), (6, 7)], n_jobs=2)
    >>> results
    [6, 20, 42]
    """
    ctx = get_context(start_method)
    with ctx.Pool(n_jobs) as pool:
        results = pool.starmap(func, iterable)
    return results

