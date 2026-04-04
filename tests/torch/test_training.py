import pytest
from rtools.torch.training import RunningStats


def test_running_stats_initialization():
    stats = RunningStats(['loss', 'accuracy'])

    assert stats.metrics == ['loss', 'accuracy']
    assert stats.total == {'loss': 0.0, 'accuracy': 0.0}
    assert stats.count == 0


def test_running_stats_update_and_mean():
    stats = RunningStats(['loss'])
    stats.update({'loss': 1.0})
    stats.update({'loss': 3.0}, elem=2)

    assert stats.total['loss'] == 7.0
    assert stats.count == 3
    assert stats.mean()['loss'] == pytest.approx(7.0 / 3.0)


def test_running_stats_unregistered_metric_raises():
    stats = RunningStats(['loss'])

    with pytest.raises(KeyError, match="'accuracy' was not registered"):
        stats.update({'accuracy': 0.9})


def test_running_stats_mean_before_update_raises():
    stats = RunningStats(['loss'])

    with pytest.raises(RuntimeError, match="before any update"):
        stats.mean()
