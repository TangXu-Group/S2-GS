"""Loss functions for standalone NDC optimization."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _weighted_mean(value: Tensor, weight: Tensor) -> Tensor:
    """Compute a numerically stable weighted mean."""
    weight_sum = weight.sum() + 1e-6
    return (value * weight).sum() / weight_sum


def _ssim_map(img_a: Tensor, img_b: Tensor) -> Tensor:
    """Compute a small-window SSIM map without external dependencies."""
    img_a_b = img_a.unsqueeze(0)
    img_b_b = img_b.unsqueeze(0)

    mu_a = F.avg_pool2d(img_a_b, kernel_size=3, stride=1, padding=1)
    mu_b = F.avg_pool2d(img_b_b, kernel_size=3, stride=1, padding=1)

    sigma_a = F.avg_pool2d(img_a_b * img_a_b, kernel_size=3, stride=1, padding=1) - mu_a * mu_a
    sigma_b = F.avg_pool2d(img_b_b * img_b_b, kernel_size=3, stride=1, padding=1) - mu_b * mu_b
    sigma_ab = F.avg_pool2d(img_a_b * img_b_b, kernel_size=3, stride=1, padding=1) - mu_a * mu_b

    c1 = 0.01 * 0.01
    c2 = 0.03 * 0.03
    numerator = (2.0 * mu_a * mu_b + c1) * (2.0 * sigma_ab + c2)
    denominator = (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a + sigma_b + c2) + 1e-6
    ssim = numerator / denominator
    return ssim.squeeze(0).mean(dim=0)


def photometric_loss(img_a: Tensor, img_b_warped: Tensor, weight: Tensor) -> Tensor:
    """
    Compute the weighted photometric loss.

    Args:
        img_a: Source image with shape ``(3, H, W)``.
        img_b_warped: Warped target image with shape ``(3, H, W)``.
        weight: Visibility weight map with shape ``(H, W)``.

    Returns:
        Scalar tensor loss.
    """
    l1 = torch.abs(img_a - img_b_warped).mean(dim=0)
    ssim_term = 1.0 - _ssim_map(img_a, img_b_warped)
    combined = 0.85 * l1 + 0.15 * ssim_term
    return _weighted_mean(combined, weight)


def geometry_loss(
    depth_pred: Tensor,
    depth_target: Tensor,
    valid_mask: Tensor,
    delta: float = 0.1,
) -> Tensor:
    """
    Compute a masked robust geometry loss against external depth supervision.

    Args:
        depth_pred: Predicted depth map with shape ``(H, W)``.
        depth_target: Geometry target depth map with shape ``(H, W)``.
        valid_mask: Boolean or float mask with shape ``(H, W)``.
        delta: Huber transition point in depth units.

    Returns:
        Scalar tensor loss.
    """
    valid_weight = valid_mask.to(dtype=depth_pred.dtype)
    depth_error = F.huber_loss(depth_pred, depth_target, reduction="none", delta=delta)
    return _weighted_mean(depth_error, valid_weight)

