"""Tests for ResNet model architectures."""

import torch
import pytest
from rtools.torch.models import (
    ResNetBlock,
    ResNet18,
    ResNet34,
    ResNetTransposeBlock,
    ResNet18Decoder,
    ResNet34Decoder,
)


def test_resnet_block_identity_shortcut_preserves_shape():
    block = ResNetBlock(in_channels=16, out_channels=16, stride=1)
    x = torch.randn(2, 16, 8, 8)

    out = block(x)

    assert out.shape == (2, 16, 8, 8)
    assert isinstance(block.shortcut, torch.nn.Sequential)
    assert len(block.shortcut) == 0


def test_resnet_block_projection_shortcut_on_channel_change():
    block = ResNetBlock(in_channels=16, out_channels=32, stride=2)
    x = torch.randn(2, 16, 8, 8)

    out = block(x)

    assert out.shape == (2, 32, 4, 4)
    assert len(block.shortcut) > 0


@pytest.mark.parametrize("model_cls", [ResNet18, ResNet34])
def test_resnet_encoder_output_shape(model_cls):
    model = model_cls(in_channels=3, n_output=10)
    x = torch.randn(2, 3, 64, 64)

    out = model(x)

    assert out.shape == (2, 10)


@pytest.mark.parametrize("model_cls", [ResNet18, ResNet34])
def test_resnet_encoder_handles_non_default_in_channels(model_cls):
    model = model_cls(in_channels=1, n_output=5)
    x = torch.randn(1, 1, 64, 64)

    out = model(x)

    assert out.shape == (1, 5)


def test_resnet_transpose_block_upsamples_with_stride():
    block = ResNetTransposeBlock(in_channels=32, out_channels=16, stride=2)
    x = torch.randn(2, 32, 4, 4)

    out = block(x)

    assert out.shape == (2, 16, 8, 8)


@pytest.mark.parametrize("decoder_cls", [ResNet18Decoder, ResNet34Decoder])
def test_resnet_decoder_output_shape(decoder_cls):
    decoder = decoder_cls(latent_dim=10, out_channels=3)
    z = torch.randn(2, 10)

    out = decoder(z)

    assert out.shape == (2, 3, 64, 64)


def test_resnet_encoder_decoder_roundtrip_shapes():
    encoder = ResNet18(in_channels=3, n_output=10)
    decoder = ResNet18Decoder(latent_dim=10, out_channels=3)
    x = torch.randn(4, 3, 64, 64)

    z = encoder(x)
    x_hat = decoder(z)

    assert x_hat.shape == x.shape
