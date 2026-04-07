"""
Unit tests for model forward pass shapes.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.resnet_baseline import ResNet50Classifier
from src.models.vit_model import ViTHistoClassifier, measure_attention_memory


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


class TestViTHistoClassifier:
    def test_forward_shape(self):
        """ViT forward pass with image_size=224, patch_size=16 -> output (B, 4)."""
        model = ViTHistoClassifier(
            num_classes=4,
            use_flash_attention=True,
            patch_size=16,
            image_size=224,
            pretrained=False,
        )
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 4)

    def test_forward_no_nan(self):
        """ViT output should not contain NaN."""
        model = ViTHistoClassifier(
            num_classes=4,
            use_flash_attention=False,
            pretrained=False,
        )
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert not torch.isnan(out).any()

    def test_forward_standard_attention(self):
        """ViT works with standard attention (no flash)."""
        model = ViTHistoClassifier(
            num_classes=4,
            use_flash_attention=False,
            pretrained=False,
        )
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 4)

    def test_get_attention_maps(self):
        """get_attention_maps returns tensor of shape (B, num_heads, N+1, N+1)."""
        model = ViTHistoClassifier(
            num_classes=4,
            use_flash_attention=True,
            patch_size=16,
            image_size=224,
            pretrained=False,
        )
        x = torch.randn(1, 3, 224, 224)
        attn = model.get_attention_maps(x)
        # num_patches = (224/16)^2 = 196, seq_len = 196 + 1 (CLS) = 197
        assert attn.shape == (1, 12, 197, 197)

    def test_count_parameters(self):
        """count_parameters returns dict with expected keys."""
        model = ViTHistoClassifier(
            num_classes=4,
            pretrained=False,
        )
        params = model.count_parameters()
        assert "total" in params
        assert "trainable" in params
        assert "patch_embed" in params
        assert "transformer_blocks" in params
        assert "classifier_head" in params
        assert params["total"] > 0
        assert params["total"] == params["trainable"]


class TestMeasureAttentionMemory:
    def test_reduction_factor_above_5(self):
        """measure_attention_memory(seq_len=1024) returns reduction_factor > 5."""
        result = measure_attention_memory(seq_len=1024)
        assert result["reduction_factor"] > 5

    def test_standard_greater_than_flash(self):
        """Standard attention should use more memory than FlashAttention."""
        result = measure_attention_memory(seq_len=512)
        assert result["standard_attention_mb"] > result["flash_attention_mb"]

    def test_keys_present(self):
        """Result dict should contain expected keys."""
        result = measure_attention_memory(seq_len=256)
        assert "standard_attention_mb" in result
        assert "flash_attention_mb" in result
        assert "reduction_factor" in result

    def test_scales_quadratically_for_standard(self):
        """Standard attention memory should scale quadratically with seq_len."""
        mem_256 = measure_attention_memory(seq_len=256)
        mem_512 = measure_attention_memory(seq_len=512)
        # Doubling seq_len should ~4x memory for standard attention
        ratio = mem_512["standard_attention_mb"] / mem_256["standard_attention_mb"]
        assert 3.9 < ratio < 4.1
