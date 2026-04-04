"""Parallel processing utilities for parameter sweeps and multiprocessing tasks."""

from .sweep import parallel_map, parallel_starmap

__all__ = ['parallel_map', 'parallel_starmap']

