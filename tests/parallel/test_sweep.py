"""Tests for parallel processing utilities."""

import pytest
from rtools.parallel import parallel_map, parallel_starmap


def square(x):
    """Simple test function."""
    return x ** 2


def multiply(x, y):
    """Test function with two arguments."""
    return x * y


def test_parallel_map_basic():
    """Test basic parallel map."""
    results = parallel_map(square, [1, 2, 3, 4], n_jobs=2)
    assert results == [1, 4, 9, 16]


def test_parallel_map_single_item():
    """Test parallel map with single item."""
    results = parallel_map(square, [5], n_jobs=1)
    assert results == [25]


def test_parallel_map_empty():
    """Test parallel map with empty iterable."""
    results = parallel_map(square, [], n_jobs=1)
    assert results == []


def test_parallel_map_order_preserved():
    """Test that parallel map preserves order."""
    results = parallel_map(square, list(range(10)), n_jobs=2)
    expected = [x ** 2 for x in range(10)]
    assert results == expected


def test_parallel_starmap_basic():
    """Test basic parallel starmap."""
    results = parallel_starmap(
        multiply,
        [(2, 3), (4, 5), (6, 7)],
        n_jobs=2
    )
    assert results == [6, 20, 42]


def test_parallel_starmap_single_item():
    """Test parallel starmap with single item."""
    results = parallel_starmap(multiply, [(3, 4)], n_jobs=1)
    assert results == [12]


def test_parallel_starmap_empty():
    """Test parallel starmap with empty iterable."""
    results = parallel_starmap(multiply, [], n_jobs=1)
    assert results == []


def test_parallel_starmap_order_preserved():
    """Test that parallel starmap preserves order."""
    args = [(x, x+1) for x in range(10)]
    results = parallel_starmap(multiply, args, n_jobs=2)
    expected = [x * (x + 1) for x in range(10)]
    assert results == expected


def test_parallel_map_no_workers_specified():
    """Test parallel map with n_jobs=None."""
    results = parallel_map(square, [1, 2, 3])
    assert results == [1, 4, 9]


def test_parallel_starmap_no_workers_specified():
    """Test parallel starmap with n_jobs=None."""
    results = parallel_starmap(multiply, [(2, 3), (4, 5)])
    assert results == [6, 20]


def test_parallel_map_large_iterable():
    """Test parallel map with larger iterable."""
    results = parallel_map(square, list(range(100)), n_jobs=4)
    expected = [x ** 2 for x in range(100)]
    assert results == expected


def test_parallel_starmap_large_iterable():
    """Test parallel starmap with larger iterable."""
    args = [(x, x) for x in range(50)]
    results = parallel_starmap(multiply, args, n_jobs=4)
    expected = [x * x for x in range(50)]
    assert results == expected


def test_parallel_map_with_start_method():
    """Test parallel map with explicit start_method."""
    results = parallel_map(square, [1, 2, 3], n_jobs=2, start_method="spawn")
    assert results == [1, 4, 9]


def test_parallel_starmap_with_start_method():
    """Test parallel starmap with explicit start_method."""
    results = parallel_starmap(
        multiply,
        [(2, 3), (4, 5)],
        n_jobs=2,
        start_method="spawn"
    )
    assert results == [6, 20]

