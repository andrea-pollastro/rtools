import os
import random
import numpy as np
import pytest
import torch

from research_utils.pytorch.random import set_seed


def test_set_seed_is_repeatable_cpu():
    set_seed(123)

    r1 = random.random()
    n1 = np.random.rand(5)
    t1 = torch.rand(5)

    set_seed(123)

    r2 = random.random()
    n2 = np.random.rand(5)
    t2 = torch.rand(5)

    assert r1 == r2
    assert np.allclose(n1, n2)
    assert torch.equal(t1, t2)


def test_set_seed_sets_torch_determinism_flags():
    set_seed(123)

    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert torch.are_deterministic_algorithms_enabled() is True


def test_cpu_only_does_not_call_cuda_seeding(monkeypatch):
    """
    Ensures set_seed() is safe on CPU-only machines by not calling torch.cuda.* seeding.
    This test forces torch.cuda.is_available() to False.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def _boom(*args, **kwargs):
        raise AssertionError("CUDA seeding should not be called when CUDA is unavailable")

    monkeypatch.setattr(torch.cuda, "manual_seed", _boom)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", _boom)

    # Should not raise
    set_seed(123)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available on this runner")
def test_set_seed_is_repeatable_on_cuda():
    # Note: This only tests RNG repeatability on CUDA, not full op determinism.
    set_seed(123)

    t1 = torch.rand(10, device="cuda")
    set_seed(123)
    t2 = torch.rand(10, device="cuda")

    assert torch.equal(t1.cpu(), t2.cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available on this runner")
def test_set_seed_calls_cuda_seeders(monkeypatch):
    """
    Verifies the CUDA seeding functions are invoked when CUDA is available.
    """
    calls = {"manual_seed": 0, "manual_seed_all": 0}
    monkeypatch.setattr(torch.cuda, "manual_seed", lambda s: calls.__setitem__("manual_seed", calls["manual_seed"] + 1))
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda s: calls.__setitem__("manual_seed_all", calls["manual_seed_all"] + 1))

    set_seed(123)

    assert calls["manual_seed"] == 1
    assert calls["manual_seed_all"] == 1
