import random
import numpy as np
import pytest
import torch
from rtools.torch.random import set_seed


def test_set_seed_repeatable_cpu():
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


def test_set_seed_handles_missing_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    set_seed(123)
