import pytest
from pathlib import Path
import tempfile
from rtools.torch.monitoring import RunningMean, History


def test_running_mean_initialization():
    stats = RunningMean(['loss', 'accuracy'])

    assert stats.metrics == ['loss', 'accuracy']
    assert stats.total == {'loss': 0.0, 'accuracy': 0.0}
    assert stats.count == 0


def test_running_mean_update_and_mean():
    stats = RunningMean(['loss'])
    stats.update({'loss': 1.0})
    stats.update({'loss': 3.0}, elem=2)

    assert stats.total['loss'] == 7.0
    assert stats.count == 3
    assert stats.mean()['loss'] == pytest.approx(7.0 / 3.0)


def test_running_mean_unregistered_metric_raises():
    stats = RunningMean(['loss'])

    with pytest.raises(KeyError, match="'accuracy' was not registered"):
        stats.update({'accuracy': 0.9})


def test_running_mean_mean_before_update_raises():
    stats = RunningMean(['loss'])

    with pytest.raises(RuntimeError, match="before any update"):
        stats.mean()


def test_history_initialization():
    history = History(['loss', 'accuracy'])

    assert history.metrics == ['loss', 'accuracy']
    assert history.history == {'loss': [], 'accuracy': []}


def test_history_update_and_get():
    history = History(['loss', 'accuracy'])
    history.update({'loss': 0.5, 'accuracy': 0.90})
    history.update({'loss': 0.4, 'accuracy': 0.92})

    assert history.get('loss') == [0.5, 0.4]
    assert history.get('accuracy') == [0.90, 0.92]


def test_history_get_all():
    history = History(['loss', 'accuracy'])
    history.update({'loss': 0.5, 'accuracy': 0.90})
    history.update({'loss': 0.4, 'accuracy': 0.92})

    all_history = history.get_all()
    assert all_history == {
        'loss': [0.5, 0.4],
        'accuracy': [0.90, 0.92]
    }


def test_history_unregistered_metric_raises():
    history = History(['loss'])

    with pytest.raises(KeyError, match="'accuracy' was not registered"):
        history.update({'accuracy': 0.9})


def test_history_get_unregistered_metric_raises():
    history = History(['loss'])

    with pytest.raises(KeyError, match="'accuracy' was not registered"):
        history.get('accuracy')


def test_history_to_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / 'history.csv'
        
        history = History(['loss', 'accuracy'])
        history.update({'loss': 0.5, 'accuracy': 0.90})
        history.update({'loss': 0.4, 'accuracy': 0.92})
        history.update({'loss': 0.3, 'accuracy': 0.94})
        
        history.to_csv(filepath)
        
        assert filepath.exists()
        
        # Read and verify CSV content
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        assert lines[0].strip() == 'loss,accuracy'
        assert lines[1].strip() == '0.5,0.9'
        assert lines[2].strip() == '0.4,0.92'
        assert lines[3].strip() == '0.3,0.94'
