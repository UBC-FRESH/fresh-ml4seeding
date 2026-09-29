"""Test evaluation metrics."""

import numpy as np
import pytest

from ml4seeding.metrics import (
    accuracy,
    evaluate_segmentation,
    foreground_mean_iou,
    iou_per_class,
    mean_iou,
)


class TestIoUPerClass:
    def test_perfect_prediction(self):
        y = np.array([[0, 1], [2, 3]])
        ious = iou_per_class(y, y)
        for c in range(4):
            assert ious[c] == 1.0

    def test_all_wrong(self):
        y_true = np.zeros((4, 4), dtype=int)
        y_pred = np.ones((4, 4), dtype=int)
        ious = iou_per_class(y_true, y_pred)
        # Class 0: no true pixels predicted as 0, IoU should be very low
        assert ious[0] < 0.01

    def test_single_class(self):
        y_true = np.array([[1, 1], [1, 1]])
        y_pred = np.array([[1, 1], [0, 0]])
        ious = iou_per_class(y_true, y_pred)
        # 2 correct out of 4 true + 2 predicted - 2 intersection = 4 union
        assert ious[1] == pytest.approx(0.5)


class TestMeanIoU:
    def test_perfect(self):
        y = np.array([[0, 1], [2, 3]])
        assert mean_iou(y, y) == 1.0


class TestForegroundMeanIoU:
    def test_excludes_background(self):
        y_true = np.array([[0, 0], [1, 2]])
        y_pred = np.array([[0, 0], [1, 2]])
        assert foreground_mean_iou(y_true, y_pred) == 1.0

    def test_with_background_errors(self):
        y_true = np.array([[0, 0], [1, 1]])
        y_pred = np.array([[1, 1], [1, 1]])
        fg = foreground_mean_iou(y_true, y_pred)
        # Class 1: intersection=2, union=4, IoU=0.5
        # Classes 2,3: no true or predicted pixels, IoU~1.0 (smoothing)
        # Foreground mean = (0.5 + 1.0 + 1.0) / 3 ≈ 0.833
        assert fg == pytest.approx(0.833, abs=0.01)


class TestAccuracy:
    def test_perfect(self):
        y = np.array([[0, 1], [2, 3]])
        assert accuracy(y, y) == 1.0

    def test_half(self):
        y_true = np.array([[0, 0], [1, 1]])
        y_pred = np.array([[0, 1], [1, 0]])
        assert accuracy(y_true, y_pred) == 0.5


class TestEvaluateSegmentation:
    def test_returns_all_metrics(self):
        y = np.array([[0, 1], [2, 3]])
        result = evaluate_segmentation(y, y)
        assert "accuracy" in result
        assert "mean_iou" in result
        assert "foreground_mean_iou" in result
        assert "iou_background" in result
        assert "iou_good" in result
        assert "iou_fair" in result
        assert "iou_poor" in result

    def test_perfect_score(self):
        y = np.array([[0, 1], [2, 3]])
        result = evaluate_segmentation(y, y)
        assert result["accuracy"] == 1.0
        assert result["mean_iou"] == 1.0
        assert result["foreground_mean_iou"] == 1.0
