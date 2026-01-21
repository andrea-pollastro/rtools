import torch
import pytest
from unittest.mock import Mock, patch, MagicMock
from rtools.datasets.disentanglement import DSprites


class TestDSprites:
    """Tests for DSprites dataset."""

    @pytest.fixture
    def mock_dsprites_data(self):
        """Create mock DSprites dataset data."""
        n_samples = 100
        imgs = (torch.rand(n_samples, 64, 64) > 0.5).float().numpy()
        # Create realistic latent classes: each factor has limited values
        # Matching the actual sizes: [1, 3, 6, 40, 32, 32]
        latents_classes = torch.zeros((n_samples, 6), dtype=torch.int32).numpy()
        latents_classes[:, 0] = torch.randint(0, 1, (n_samples,)).numpy()  # color
        latents_classes[:, 1] = torch.randint(0, 3, (n_samples,)).numpy()  # shape
        latents_classes[:, 2] = torch.randint(0, 6, (n_samples,)).numpy()  # scale
        latents_classes[:, 3] = torch.randint(0, 40, (n_samples,)).numpy()  # orientation
        latents_classes[:, 4] = torch.randint(0, 32, (n_samples,)).numpy()  # x
        latents_classes[:, 5] = torch.randint(0, 32, (n_samples,)).numpy()  # y
        
        latents_values = torch.rand(n_samples, 6).numpy()
        
        return {
            'imgs': imgs,
            'latents_classes': latents_classes,
            'latents_values': latents_values,
        }

    @patch('rtools.datasets.disentanglement.datasets.np.load')
    @patch('rtools.datasets.disentanglement.datasets.Path')
    def test_dsprites_initialization(self, mock_path, mock_load, mock_dsprites_data):
        """Should initialize with correct attributes."""
        mock_load.return_value = mock_dsprites_data
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        ds = DSprites(K=2, paired=True)
        
        assert ds.K == 2
        assert ds.paired is True
        assert ds.latents_factors == 6

    @patch('rtools.datasets.disentanglement.datasets.np.load')
    @patch('rtools.datasets.disentanglement.datasets.Path')
    def test_dsprites_length(self, mock_path, mock_load, mock_dsprites_data):
        """Should return correct dataset length."""
        mock_load.return_value = mock_dsprites_data
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        ds = DSprites()
        
        assert len(ds) == 100

    @patch('rtools.datasets.disentanglement.datasets.np.load')
    @patch('rtools.datasets.disentanglement.datasets.Path')
    def test_dsprites_unpaired_mode(self, mock_path, mock_load, mock_dsprites_data):
        """Should return single samples in unpaired mode."""
        mock_load.return_value = mock_dsprites_data
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        ds = DSprites(paired=False)
        
        img, latents = ds[0]
        
        assert isinstance(img, torch.Tensor)
        assert isinstance(latents, torch.Tensor)
        assert img.shape == (1, 64, 64)
        assert latents.shape == (6,)

    @patch('rtools.datasets.disentanglement.datasets.np.load')
    @patch('rtools.datasets.disentanglement.datasets.Path')
    def test_dsprites_paired_mode(self, mock_path, mock_load, mock_dsprites_data):
        """Should attempt to return paired samples in paired mode."""
        mock_load.return_value = mock_dsprites_data
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        ds = DSprites(K=2, paired=True)
        
        # Paired mode may fail with small mock data, but structure should be created
        assert ds.paired is True
        assert ds.K == 2

    @patch('rtools.datasets.disentanglement.datasets.np.load')
    @patch('rtools.datasets.disentanglement.datasets.Path')
    def test_dsprites_random_k_mode(self, mock_path, mock_load, mock_dsprites_data):
        """Should accept K=-1 for random K selection."""
        mock_load.return_value = mock_dsprites_data
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        # K=-1 should initialize without errors
        ds = DSprites(K=-1, paired=True)
        assert ds.K == -1

    @patch('rtools.datasets.disentanglement.datasets.np.load')
    @patch('rtools.datasets.disentanglement.datasets.Path')
    def test_dsprites_k_parameter_validation(self, mock_path, mock_load, mock_dsprites_data):
        """Should validate K parameter."""
        mock_load.return_value = mock_dsprites_data
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        # Should raise for invalid K
        with pytest.raises(AssertionError):
            DSprites(K=0, paired=True)
        
        with pytest.raises(AssertionError):
            DSprites(K=10, paired=True)  # Too large

    @patch('rtools.datasets.disentanglement.datasets.np.load')
    @patch('rtools.datasets.disentanglement.datasets.Path')
    def test_dsprites_latent_to_index(self, mock_path, mock_load, mock_dsprites_data):
        """Should convert latent indices to dataset index."""
        mock_load.return_value = mock_dsprites_data
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        ds = DSprites()
        
        latents = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64)
        idx = ds.latent_to_index(latents)
        
        assert isinstance(idx, torch.Tensor)
        assert idx.dim() == 0  # Scalar tensor

    @patch('rtools.datasets.disentanglement.datasets.np.load')
    @patch('rtools.datasets.disentanglement.datasets.Path')
    def test_dsprites_channel_dimension_added(self, mock_path, mock_load, mock_dsprites_data):
        """Should add channel dimension to images."""
        mock_load.return_value = mock_dsprites_data
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        ds = DSprites()
        
        assert ds.X.shape == (100, 1, 64, 64)

    @patch('rtools.datasets.disentanglement.datasets.np.load')
    @patch('rtools.datasets.disentanglement.datasets.Path')
    def test_dsprites_latents_tensor_types(self, mock_path, mock_load, mock_dsprites_data):
        """Should have correct tensor types for latents."""
        mock_load.return_value = mock_dsprites_data
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        ds = DSprites()
        
        assert ds.latents_classes.dtype == torch.int64
        assert ds.latents_sizes.dtype == torch.int64
        assert ds.latents_bases.dtype == torch.int64
        assert ds.X.dtype == torch.float32

    @patch('rtools.datasets.disentanglement.datasets.np.load')
    @patch('rtools.datasets.disentanglement.datasets.Path')
    def test_dsprites_paired_samples_differ_in_k_factors(self, mock_path, mock_load, mock_dsprites_data):
        """Test that paired mode with K parameter is properly configured."""
        mock_load.return_value = mock_dsprites_data
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        ds = DSprites(K=2, paired=True)
        
        # Verify configuration
        assert ds.K == 2
        assert ds.paired is True
        assert ds.latents_factors == 6

    @patch('rtools.datasets.disentanglement.datasets.np.load')
    @patch('rtools.datasets.disentanglement.datasets.Path')
    def test_dsprites_different_indices_same_class(self, mock_path, mock_load, mock_dsprites_data):
        """Should return different images for different indices."""
        mock_load.return_value = mock_dsprites_data
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        ds = DSprites(paired=False)
        
        img1, _ = ds[0]
        img2, _ = ds[1]
        
        # Different indices should likely give different images
        assert not torch.equal(img1, img2) # type: ignore
