"""Delta-map parameterizations for standalone NDC optimization."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DecomposedDeltaMap(nn.Module):
    """
    Decompose a delta map into low- and high-frequency components.

    The low-frequency branch is represented by a sparse control-point grid that
    is bilinearly upsampled to the full image resolution. The high-frequency
    branch is a dense pixel-wise residual map.
    """

    def __init__(self, height: int, width: int, grid_h: int = 4, grid_w: int = 4) -> None:
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)

        self.control_points = nn.Parameter(torch.zeros(1, 1, self.grid_h, self.grid_w, dtype=torch.float32))
        self.pixel_delta = nn.Parameter(torch.zeros(1, 1, self.height, self.width, dtype=torch.float32))

    def _delta_low(self) -> Tensor:
        return F.interpolate(
            self.control_points,
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=True,
        ).squeeze(0).squeeze(0)

    def _delta_high(self) -> Tensor:
        return self.pixel_delta.squeeze(0).squeeze(0)

    def forward(self) -> Tensor:
        """Return the full-resolution delta map with shape ``(H, W)``."""
        return self._delta_low() + self._delta_high()

    def get_components(self) -> tuple[Tensor, Tensor]:
        """Return ``(delta_low, delta_high)`` for debugging or custom losses."""
        return self._delta_low(), self._delta_high()
