from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def _to_tensor(data: Tensor | np.ndarray, device: torch.device | str) -> Tensor:
    """Convert input data to a float32 tensor on the requested device."""
    if isinstance(data, Tensor):
        return data.to(device=device, dtype=torch.float32)
    return torch.as_tensor(data, dtype=torch.float32, device=device)


def _make_pixel_grid(height: int, width: int, device: torch.device | str) -> Tuple[Tensor, Tensor]:
    """Create pixel coordinate grids in image space."""
    ys = torch.arange(height, dtype=torch.float32, device=device)
    xs = torch.arange(width, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return grid_x, grid_y


def _normalize_grid(u: Tensor, v: Tensor, width: int, height: int) -> Tensor:
    """Convert pixel coordinates to grid_sample coordinates."""
    u_norm = 2.0 * u / max(width - 1, 1) - 1.0
    v_norm = 2.0 * v / max(height - 1, 1) - 1.0
    return torch.stack([u_norm, v_norm], dim=-1)


def _sample_map(map_data: Tensor, u: Tensor, v: Tensor) -> Tensor:
    """Bilinearly sample a ``(C,H,W)`` or ``(H,W)`` tensor at pixel coordinates."""
    if map_data.ndim == 2:
        map_data = map_data.unsqueeze(0)
    if map_data.ndim != 3:
        raise ValueError("map_data must have shape (H, W) or (C, H, W)")

    _, height, width = map_data.shape
    grid = _normalize_grid(u, v, width, height).unsqueeze(0)
    sampled = F.grid_sample(
        map_data.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.squeeze(0)


def warp_depth_to_view(
    depth_a: Tensor | np.ndarray,
    img_b: Tensor | np.ndarray,
    intrinsics_a: Dict[str, float],
    intrinsics_b: Dict[str, float],
    c2w_a: Tensor | np.ndarray,
    w2c_b: Tensor | np.ndarray,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Warp view ``B`` into the pixel grid of view ``A`` using depth from ``A``.

    Args:
        depth_a: Depth map from source view A with shape ``(H, W)``.
        img_b: Target view image with shape ``(3, H, W)``.
        intrinsics_a: Source camera intrinsics dict with ``fx``, ``fy``, ``cx``, ``cy``.
        intrinsics_b: Target camera intrinsics dict with ``fx``, ``fy``, ``cx``, ``cy``.
        c2w_a: Source camera-to-world transform with shape ``(4, 4)``.
        w2c_b: Target world-to-camera transform with shape ``(4, 4)``.

    Returns:
        Tuple ``(img_b_warped, z_b, u_b, v_b)`` where the warped image has shape
        ``(3, H, W)`` and the others have shape ``(H, W)``.
    """
    if not isinstance(intrinsics_a, dict) or not isinstance(intrinsics_b, dict):
        raise TypeError("intrinsics_a and intrinsics_b must be dictionaries")

    device = depth_a.device if isinstance(depth_a, Tensor) else (
        img_b.device if isinstance(img_b, Tensor) else "cpu"
    )
    depth_a_t = _to_tensor(depth_a, device).clamp(min=0.01, max=1000.0)
    img_b_t = _to_tensor(img_b, device)
    c2w_a_t = _to_tensor(c2w_a, device)
    w2c_b_t = _to_tensor(w2c_b, device)

    height, width = depth_a_t.shape
    grid_x, grid_y = _make_pixel_grid(height, width, device)

    fx_a = float(intrinsics_a["fx"])
    fy_a = float(intrinsics_a["fy"])
    cx_a = float(intrinsics_a["cx"])
    cy_a = float(intrinsics_a["cy"])
    fx_b = float(intrinsics_b["fx"])
    fy_b = float(intrinsics_b["fy"])
    cx_b = float(intrinsics_b["cx"])
    cy_b = float(intrinsics_b["cy"])

    z_cam = depth_a_t
    x_cam = (grid_x - cx_a) / (fx_a + 1e-6) * z_cam
    y_cam = (grid_y - cy_a) / (fy_a + 1e-6) * z_cam

    ones = torch.ones_like(z_cam)
    cam_points = torch.stack([x_cam, y_cam, z_cam, ones], dim=-1).reshape(-1, 4)
    world_points = (c2w_a_t @ cam_points.t()).t()
    cam_b = (w2c_b_t @ world_points.t()).t()

    x_b = cam_b[:, 0].reshape(height, width)
    y_b = cam_b[:, 1].reshape(height, width)
    z_b = cam_b[:, 2].reshape(height, width)

    u_b = fx_b * x_b / (z_b + 1e-6) + cx_b
    v_b = fy_b * y_b / (z_b + 1e-6) + cy_b

    img_b_warped = _sample_map(img_b_t, u_b, v_b)
    return img_b_warped, z_b, u_b, v_b
