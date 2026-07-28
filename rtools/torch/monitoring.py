from typing import List, Dict
import csv
from pathlib import Path


class RunningMean:
    """
    Track running mean of multiple metrics.

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
            raise RuntimeError("RunningMean.mean() called before any update.")

        return {
            m: self.total[m] / self.count
            for m in self.metrics
        }


class History:
    """
    Track history of metrics as lists for visualization and analysis.

    Parameters
    ----------
    metrics : List[str]
        List of metric names to track.

    Examples
    --------
    >>> history = History(['loss', 'accuracy'])
    >>> history.update({'loss': 0.5, 'accuracy': 0.90})
    >>> history.update({'loss': 0.4, 'accuracy': 0.92})
    >>> history.to_csv('training_history.csv')
    """

    def __init__(self, metrics: List[str]):
        self.metrics = metrics
        self.history: Dict[str, List[float]] = {m: [] for m in metrics}

    def update(self, values: Dict[str, float]) -> None:
        """
        Append metric values to history.

        Parameters
        ----------
        values : Dict[str, float]
            Dictionary mapping metric names to values to append.

        Raises
        ------
        KeyError
            If a metric name was not registered in `__init__`.
        """
        for m, v in values.items():
            if m not in self.history:
                raise KeyError(f"Metric '{m}' was not registered.")
            self.history[m].append(v)

    def get(self, metric: str) -> List[float]:
        """
        Get history for a single metric.

        Parameters
        ----------
        metric : str
            Metric name.

        Returns
        -------
        List[float]
            List of recorded values for this metric.

        Raises
        ------
        KeyError
            If metric was not registered.
        """
        if metric not in self.history:
            raise KeyError(f"Metric '{metric}' was not registered.")
        return self.history[metric].copy()

    def get_all(self) -> Dict[str, List[float]]:
        """
        Get history for all metrics.

        Returns
        -------
        Dict[str, List[float]]
            Dictionary mapping metric names to their value lists.
        """
        return {m: values.copy() for m, values in self.history.items()}

    def to_csv(self, filepath: str | Path) -> None:
        """
        Export history to CSV file.

        Parameters
        ----------
        filepath : str or Path
            Path where CSV file will be saved.
        """
        filepath = Path(filepath)
        
        # Get the maximum history length
        max_length = max((len(vals) for vals in self.history.values()), default=0)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.metrics)
            writer.writeheader()
            
            for i in range(max_length):
                row = {}
                for metric in self.metrics:
                    if i < len(self.history[metric]):
                        row[metric] = self.history[metric][i]
                    else:
                        row[metric] = ''
                writer.writerow(row)

                