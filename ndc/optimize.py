"""Standalone optimization loop for geometry- and photometric-supervised RRM depth correction."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from .delta_map import DecomposedDeltaMap
from .loss import geometry_loss, photometric_loss
from .warp import warp_depth_to_view


GEOMETRY_HUBER_DELTA = 0.1
DELTA_GRID_H = 8
DELTA_GRID_W = 8
DELTA_HIGH_REG = 5e-3
LOW_FREQ_LR_SCALE = 20.0


def _to_tensor(data: Tensor | np.ndarray, device: torch.device) -> Tensor:
    """Convert numpy arrays or tensors to float32 tensors on the target device."""
    if isinstance(data, Tensor):
        return data.to(device=device, dtype=torch.float32)
    return torch.as_tensor(data, dtype=torch.float32, device=device)


def run_ndc(
    depths_aligned: List[Tensor | np.ndarray],
    geometry_depths: List[Tensor | np.ndarray],
    images: List[Tensor | np.ndarray],
    intrinsics: List[Dict[str, float]],
    c2w_list: List[np.ndarray | Tensor],
    w2c_list: List[np.ndarray | Tensor],
    n_iters: int = 500,
    lr: float = 1e-3,
    lambda_geom: float = 0.1,
    lambda_photo: float = 1.0,
    device: str = "cuda",
) -> Tuple[List[np.ndarray], List[Dict[str, np.ndarray]]]:
    
    n_views = len(depths_aligned)
    if not (
        len(geometry_depths) == len(images) == len(intrinsics) == len(c2w_list) == len(w2c_list) == n_views
    ):
        raise ValueError("run_ndc expects all inputs to match the number of input depths")
    if n_views < 2:
        raise ValueError("run_ndc expects at least two views")

    device_t = torch.device(device)
    depths_t = [_to_tensor(depth, device_t).clamp(min=0.01, max=1000.0) for depth in depths_aligned]
    geometry_t = [_to_tensor(depth, device_t).clamp(min=0.0, max=1000.0) for depth in geometry_depths]
    images_t = []
    for image in images:
        image_t = _to_tensor(image, device_t)
        if image_t.ndim != 3:
            raise ValueError("images must have shape (3, H, W) or (H, W, 3)")
        if image_t.shape[0] != 3 and image_t.shape[-1] == 3:
            image_t = image_t.permute(2, 0, 1)
        if image_t.shape[0] != 3:
            raise ValueError("images must have three channels")
        images_t.append(image_t)
    c2w_t = [_to_tensor(mat, device_t) for mat in c2w_list]
    w2c_t = [_to_tensor(mat, device_t) for mat in w2c_list]
    height, width = depths_t[0].shape
    for depth in depths_t:
        if depth.shape != (height, width):
            raise ValueError("all depth maps must share the same shape")
    for depth in geometry_t:
        if depth.shape != (height, width):
            raise ValueError("all geometry depth maps must have shape (H, W)")
    for image in images_t:
        if image.shape[1:] != (height, width):
            raise ValueError("all images must have shape (3, H, W) matching the depth maps")

    delta_modules = nn.ModuleList(
        [DecomposedDeltaMap(height, width, grid_h=DELTA_GRID_H, grid_w=DELTA_GRID_W).to(device_t) for _ in range(n_views)]
    )
    optimizer = Adam(
        [
            {"params": [module.control_points for module in delta_modules], "lr": lr * LOW_FREQ_LR_SCALE},
            {"params": [module.pixel_delta for module in delta_modules], "lr": lr},
        ]
    )
    scheduler = ExponentialLR(optimizer, gamma=0.995)
    directed_pairs = [(src, dst) for src in range(n_views) for dst in range(n_views) if src != dst]
    for _ in range(n_iters):
        optimizer.zero_grad()

        corrected_depths_clamped = []
        delta_high_components = []
        for depth_in, delta_module in zip(depths_t, delta_modules):
            delta_map = delta_module()
            _, delta_high = delta_module.get_components()
            corrected_depths_clamped.append((depth_in + delta_map).clamp(min=0.01, max=1000.0))
            delta_high_components.append(delta_high)

        total_loss = torch.zeros((), dtype=torch.float32, device=device_t)
        for depth_clamped, depth_geom, delta_high in zip(corrected_depths_clamped, geometry_t, delta_high_components):
            if lambda_geom > 0.0:
                geom_valid = depth_geom > 0.0
                total_loss = total_loss + lambda_geom * geometry_loss(
                    depth_clamped,
                    depth_geom,
                    geom_valid,
                    delta=GEOMETRY_HUBER_DELTA,
                )
            total_loss = total_loss + DELTA_HIGH_REG * (delta_high * delta_high).mean()
        if lambda_photo > 0.0:
            for src_idx, dst_idx in directed_pairs:
                img_b_warped, z_b, u_b, v_b = warp_depth_to_view(
                    depth_a=corrected_depths_clamped[src_idx],
                    img_b=images_t[dst_idx],
                    intrinsics_a=intrinsics[src_idx],
                    intrinsics_b=intrinsics[dst_idx],
                    c2w_a=c2w_t[src_idx],
                    w2c_b=w2c_t[dst_idx],
                )
                in_bounds = (u_b >= 0.0) & (u_b <= (width - 1)) & (v_b >= 0.0) & (v_b <= (height - 1))
                valid_weight = ((z_b > 0.0) & in_bounds).to(dtype=corrected_depths_clamped[src_idx].dtype)
                total_loss = total_loss + lambda_photo * photometric_loss(
                    images_t[src_idx],
                    img_b_warped,
                    valid_weight,
                )

        total_loss.backward()
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        results = []
        components = []
        for depth_in, delta_module in zip(depths_t, delta_modules):
            delta_low, delta_high = delta_module.get_components()
            delta_total = delta_low + delta_high
            results.append((depth_in + delta_total).clamp(min=0.01, max=1000.0).detach().cpu().numpy())
            components.append(
                {
                    "delta_low": delta_low.detach().cpu().numpy(),
                    "delta_high": delta_high.detach().cpu().numpy(),
                    "delta_total": delta_total.detach().cpu().numpy(),
                }
            )
    return results, components

