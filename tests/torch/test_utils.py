"""Tests for utility functions."""

import torch
import pytest
from rtools.torch.utils import to_device


def test_to_device_single_tensor():
    """Test moving a single tensor to device."""
    x = torch.randn(3, 4)
    result = to_device(x, 'cpu')
    assert result.device.type == 'cpu'
    assert torch.equal(result, x.cpu())


def test_to_device_tuple_of_tensors():
    """Test moving tuple of tensors to device."""
    x = torch.randn(3, 4)
    y = torch.randn(2, 5)
    result = to_device((x, y), 'cpu')
    
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0].device.type == 'cpu'
    assert result[1].device.type == 'cpu'


def test_to_device_list_of_tensors():
    """Test moving list of tensors to device."""
    x = torch.randn(3, 4)
    y = torch.randn(2, 5)
    result = to_device([x, y], 'cpu')
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].device.type == 'cpu'
    assert result[1].device.type == 'cpu'


def test_to_device_dict_of_tensors():
    """Test moving dict of tensors to device."""
    x = torch.randn(3, 4)
    y = torch.randn(2, 5)
    result = to_device({'a': x, 'b': y}, 'cpu')
    
    assert isinstance(result, dict)
    assert result['a'].device.type == 'cpu'
    assert result['b'].device.type == 'cpu'


def test_to_device_nested_structure():
    """Test moving nested structure of tensors."""
    x = torch.randn(3, 4)
    y = torch.randn(2, 5)
    data = {
        'tensors': (x, y),
        'list': [x, y],
        'single': x,
    }
    result = to_device(data, 'cpu')
    
    assert result['tensors'][0].device.type == 'cpu'
    assert result['tensors'][1].device.type == 'cpu'
    assert result['list'][0].device.type == 'cpu'
    assert result['list'][1].device.type == 'cpu'
    assert result['single'].device.type == 'cpu'


def test_to_device_mixed_data_types():
    """Test with mixed tensors and non-tensors."""
    x = torch.randn(3, 4)
    data = {
        'tensor': x,
        'number': 42,
        'string': 'hello',
        'list': [x, 'mixed', 3.14],
    }
    result = to_device(data, 'cpu')
    
    assert result['tensor'].device.type == 'cpu'
    assert result['number'] == 42
    assert result['string'] == 'hello'
    assert result['list'][0].device.type == 'cpu'
    assert result['list'][1] == 'mixed'
    assert result['list'][2] == 3.14


def test_to_device_preserves_tensor_values():
    """Test that values are preserved when moving to device."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = to_device(x, 'cpu')
    assert torch.equal(result, x)


def test_to_device_torch_device_object():
    """Test with torch.device object as input."""
    x = torch.randn(3, 4)
    device = torch.device('cpu')
    result = to_device(x, device)
    assert result.device == device
