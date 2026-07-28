import torch
import pytest
from unittest.mock import MagicMock, patch
from rtools.datasets.disentanglement import DSprites


@pytest.fixture
def mock_dsprites_data():
    n_samples = 100
    imgs = (torch.rand(n_samples, 64, 64) > 0.5).float().numpy()
    latents_classes = torch.randint(0, 6, (n_samples, 6), dtype=torch.int64).numpy()
    latents_values = torch.rand(n_samples, 6).numpy()

    return {
        'imgs': imgs,
        'latents_classes': latents_classes,
        'latents_values': latents_values,
    }


def test_dsprites_initialization(mock_dsprites_data):
    with patch('rtools.datasets.disentanglement.datasets.np.load', return_value=mock_dsprites_data), \
         patch('rtools.datasets.disentanglement.datasets.Path') as mock_path:
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        ds = DSprites(K=2, paired=True)

        assert ds.K == 2
        assert ds.paired is True
        assert ds.latents_factors == 6


def test_dsprites_length(mock_dsprites_data):
    with patch('rtools.datasets.disentanglement.datasets.np.load', return_value=mock_dsprites_data), \
         patch('rtools.datasets.disentanglement.datasets.Path') as mock_path:
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        ds = DSprites()
        assert len(ds) == 100


def test_dsprites_unpaired_mode_returns_single_sample(mock_dsprites_data):
    with patch('rtools.datasets.disentanglement.datasets.np.load', return_value=mock_dsprites_data), \
         patch('rtools.datasets.disentanglement.datasets.Path') as mock_path:
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        ds = DSprites(paired=False)
        img, latent_classes, latent_values = ds[0]

        assert img.shape == (1, 64, 64)
        assert latent_classes.shape == (6,)
        assert latent_values.shape == (6,)


def test_dsprites_k_parameter_validation(mock_dsprites_data):
    with patch('rtools.datasets.disentanglement.datasets.np.load', return_value=mock_dsprites_data), \
         patch('rtools.datasets.disentanglement.datasets.Path') as mock_path:
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        with pytest.raises(AssertionError):
            DSprites(K=0, paired=True)

        with pytest.raises(AssertionError):
            DSprites(K=10, paired=True)
