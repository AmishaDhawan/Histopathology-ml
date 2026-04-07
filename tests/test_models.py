"""
Unit tests for model forward pass shapes.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.resnet_baseline import ResNet50Classifier


class TestResNet50Classifier:
    def test_forward_shape(self):
        """Output shape should be (batch_size, num_classes)."""
        model = ResNet50Classifier(num_classes=4, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 4)

    def test_forward_no_nan(self):
        """Output should not contain NaN."""
        model = ResNet50Classifier(num_classes=4, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert not torch.isnan(out).any()

    def test_custom_num_classes(self):
        """Should support different numbers of classes."""
        model = ResNet50Classifier(num_classes=10, pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 10)
