"""
Lp-convolutional network support for the GUI / HPC framework.
This module is **self-contained** and makes no assumptions about
global image size – it derives the fully-connected input dimension
from the `input_size` you pass to `LpCNN`.

Import path:  from src.lpconv import LpCNN
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Lp-convolution building-block
# ──────────────────────────────────────────────────────────────────────────────
class LpConv2d(nn.Module):
    """
    A convolutional layer that multiplies its weights with a learned
    multivariate-p normal mask (MPND) as described in the original paper.
    The implementation is channel-wise: every **out_channel** has its own
    p-value and covariance.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int],
        stride: int = 1,
        padding: int = 0,
        p_init: float = 2.0,
        sigma_init: float | None = None,
    ):
        super().__init__()
        k = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.kernel_size = k
        self.stride = stride
        self.padding = padding

        # Standard conv weight
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *k))
        
        # Learned parameters of the MPND mask
        self.p = nn.Parameter(torch.full((out_channels,), float(p_init)))
        if sigma_init is None:
            sigma_init = k[0] / 2.0
        self.L_diag = nn.Parameter(torch.full((out_channels, 2), 1.0 / sigma_init))
        self.L_offdiag = nn.Parameter(torch.zeros(out_channels, 1))

        # Pre-compute offsets grid once
        self.register_buffer("offsets", self._create_offset_grid())

    # ------------------------------------------------------------------
    def _create_offset_grid(self) -> torch.Tensor:
        kh, kw = self.kernel_size
        h = torch.linspace(0, kh - 1, kh) - (kh - 1) / 2
        w = torch.linspace(0, kw - 1, kw) - (kw - 1) / 2
        h, w = torch.meshgrid(h, w, indexing="ij")
        off = torch.stack([h, w], dim=0)          # [2, kh, kw]
        # Normalise to [0,1] for numerical stability
        off = (off - off.min()) / (off.max() - off.min())
        return off.unsqueeze(0)                   # [1, 2, kh, kw]

    def _build_covariance_matrix(self) -> torch.Tensor:
        """Cholesky-like construction of positive-definite Σ from diag/off-diag."""
        L = torch.zeros(self.weight.size(0), 2, 2, device=self.weight.device)
        L[:, 0, 0] = self.L_diag[:, 0]
        L[:, 1, 1] = self.L_diag[:, 1]
        L[:, 1, 0] = self.L_offdiag[:, 0]
        return torch.bmm(L, L.transpose(1, 2))     # Σ = L·Lᵀ

    def _compute_mask(self) -> torch.Tensor:
        p = torch.clamp(self.p, min=1.0)
        Σ = self._build_covariance_matrix()        # [O,2,2]
        off = self.offsets.expand(Σ.size(0), -1, -1, -1)  # [O,2,kh,kw]
        Σδ = torch.einsum("oij,ojhw->oihw", Σ, off)       # [O,2,kh,kw]
        lp_norm = (Σδ.abs() ** p.view(-1, 1, 1, 1)).sum(dim=1) ** (1.0 / p.view(-1, 1, 1))
        return torch.exp(-lp_norm).unsqueeze(1)    # [O,1,kh,kw]

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self._compute_mask()
        w_eff = self.weight * mask                 # masked weights
        return F.conv2d(x, w_eff, stride=self.stride, padding=self.padding)
# ──────────────────────────────────────────────────────────────────────────────


class LpCNN(nn.Module):
    """
    Two-stage CNN composed of **LpConv2d** layers.

    Parameters
    ----------
    num_classes : int
        Number of output classes (defaults to 4 – the 4-class “simple” task).
    input_size : int
        Height/width of the square input images. Must be divisible by 4
        because we apply two 2×2 MaxPools.
    dropout : float
        Drop-out probability before the final FC layer.
    in_channels : int
        Number of channels in the input images.  
        • 1  → grayscale MRI slices (GUI default)  
        • 3  → RGB images
    """
    def __init__(
        self,
        num_classes: int = 4,
        input_size: int = 32,
        dropout: float = 0.0,
        in_channels: int = 1          # NEW
    ):
        super().__init__()
        if input_size % 4 != 0:
            raise ValueError(
                "LpCNN expects input_size divisible by 4 (because of two 2×2 pools)."
            )

        # -------- feature extractor --------
        self.conv1 = LpConv2d(in_channels, 64, kernel_size=7, padding=3)   # ← CHANGED
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = LpConv2d(64, 128, kernel_size=7, padding=3)
        self.bn2   = nn.BatchNorm2d(128)
        self.relu  = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool2d(2, 2)
        self.drop  = nn.Dropout(dropout)

        # -------- classifier --------
        flat_dim = 128 * (input_size // 4) * (input_size // 4)
        self.fc = nn.Linear(flat_dim, num_classes)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.flatten(1)
        x = self.drop(x)
        return self.fc(x)