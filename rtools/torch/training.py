from typing import List, Dict

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
        self.total_sq: Dict[str, float] = {m: 0.0 for m in self.metrics}
        self.count = 0

    def update(self, values: Dict[str, float], elem: int = 1) -> None:
        """
        Update running totals with new metric values.

        Parameters
        ----------
        values : Dict[str, float]
            Dictionary mapping metric names to their new values.
        elem : int, optional
            Weight for this update. Default is 1.

        Raises
        ------
        KeyError
            If a metric name in `values` was not registered in `__init__`.
        """
        for m, v in values.items():
            if m not in self.total:
                raise KeyError(f"Metric '{m}' was not registered.")
            self.total[m] += v * elem
            self.total_sq[m] += (v ** 2) * elem
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
    
    def std(self) -> Dict[str, float]:
        """
        Compute the running standard deviation equivalent to torch.std
        (unbiased estimator, Bessel correction).

        Returns
        -------
        Dict[str, float]
            Dictionary mapping metric names to their standard deviations.

        Raises
        ------
        RuntimeError
            If called before at least two updates.
        """
        if self.count < 2:
            raise RuntimeError(
                "RunningStats.std() requires at least two samples "
                "(same behavior as torch.std with unbiased=True)."
            )

        mean = self.mean()
        n = self.count

        return {
            m: ((self.total_sq[m] - n * mean[m] ** 2) / (n - 1)) ** 0.5
            for m in self.metrics
        }