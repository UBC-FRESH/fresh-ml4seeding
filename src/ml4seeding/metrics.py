"""Evaluation metrics: per-class IoU, mean IoU, foreground mean IoU."""

from __future__ import annotations

import numpy as np


def iou_per_class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 4,
    smooth: float = 1e-7,
) -> dict[int, float]:
    """Compute Intersection-over-Union for each class.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth class labels, shape (H, W) or (N, H, W).
    y_pred : np.ndarray
        Predicted class labels, shape (H, W) or (N, H, W).
    num_classes : int
        Number of classes.
    smooth : float
        Smoothing constant to avoid division by zero.

    Returns
    -------
    dict mapping class_id -> IoU value.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    ious = {}
    for c in range(num_classes):
        true_c = (y_true == c).astype(np.float64)
        pred_c = (y_pred == c).astype(np.float64)

        intersection = np.sum(true_c * pred_c)
        union = np.sum(true_c) + np.sum(pred_c) - intersection

        ious[c] = float((intersection + smooth) / (union + smooth))

    return ious


def mean_iou(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 4,
    smooth: float = 1e-7,
) -> float:
    """Compute mean IoU across all classes."""
    ious = iou_per_class(y_true, y_pred, num_classes, smooth)
    return float(np.mean(list(ious.values())))


def foreground_mean_iou(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 4,
    background_class: int = 0,
    smooth: float = 1e-7,
) -> float:
    """Compute mean IoU excluding the background class."""
    ious = iou_per_class(y_true, y_pred, num_classes, smooth)
    fg_ious = [v for k, v in ious.items() if k != background_class]
    return float(np.mean(fg_ious))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute overall pixel accuracy."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(y_true == y_pred))


def evaluate_segmentation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 4,
    class_names: dict[int, str] | None = None,
) -> dict[str, float]:
    """Compute all evaluation metrics.

    Returns a dict with accuracy, per-class IoU, mean IoU, and foreground mean IoU.
    """
    if class_names is None:
        class_names = {0: "background", 1: "good", 2: "fair", 3: "poor"}

    ious = iou_per_class(y_true, y_pred, num_classes)

    result = {
        "accuracy": accuracy(y_true, y_pred),
        "mean_iou": mean_iou(y_true, y_pred, num_classes),
        "foreground_mean_iou": foreground_mean_iou(y_true, y_pred, num_classes),
    }

    for c, name in class_names.items():
        result[f"iou_{name}"] = ious[c]

    return result
