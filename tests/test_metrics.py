"""
Unit tests for metrics: macro-F1 and AUROC.

Tests:
- Known inputs produce expected outputs
- Perfect predictions return F1=1.0
- All-wrong predictions return F1=0.0
- AUROC returns valid values
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.training.metrics import compute_macro_f1, compute_auroc


class TestMacroF1:
    def test_perfect_predictions(self):
        """Perfect predictions should return F1=1.0."""
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        f1 = compute_macro_f1(y_true, y_pred)
        assert f1 == pytest.approx(1.0)

    def test_all_wrong_predictions(self):
        """All wrong predictions should return F1=0.0."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        f1 = compute_macro_f1(y_true, y_pred)
        assert f1 == pytest.approx(0.0)

    def test_known_input(self):
        """Test with a known input where we can compute F1 manually."""
        # Class 0: precision=2/2=1.0, recall=2/3=0.667, F1=0.8
        # Class 1: precision=2/3=0.667, recall=2/2=1.0, F1=0.8
        # Macro F1 = (0.8 + 0.8) / 2 = 0.8
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1])
        f1 = compute_macro_f1(y_true, y_pred)
        assert f1 == pytest.approx(0.8, abs=0.01)

    def test_single_class(self):
        """All same class should return F1=1.0 when predictions match."""
        y_true = np.array([2, 2, 2, 2])
        y_pred = np.array([2, 2, 2, 2])
        f1 = compute_macro_f1(y_true, y_pred)
        assert f1 == pytest.approx(1.0)

    def test_returns_float(self):
        """Should return a float value."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        f1 = compute_macro_f1(y_true, y_pred)
        assert isinstance(f1, float)


class TestAUROC:
    def test_perfect_scores(self):
        """Perfect probability scores should return AUROC near 1.0."""
        y_true = np.array([0, 1, 2, 3])
        y_scores = np.eye(4)  # Perfect one-hot predictions
        auroc = compute_auroc(y_true, y_scores, num_classes=4)
        assert auroc == pytest.approx(1.0)

    def test_returns_float(self):
        """Should return a float value."""
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        y_scores = np.random.rand(8, 4)
        y_scores = y_scores / y_scores.sum(axis=1, keepdims=True)
        auroc = compute_auroc(y_true, y_scores, num_classes=4)
        assert isinstance(auroc, float)

    def test_auroc_range(self):
        """AUROC should be between 0 and 1."""
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        y_scores = np.random.rand(8, 4)
        y_scores = y_scores / y_scores.sum(axis=1, keepdims=True)
        auroc = compute_auroc(y_true, y_scores, num_classes=4)
        assert 0.0 <= auroc <= 1.0
