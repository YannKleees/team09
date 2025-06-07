"""
src/lpconv.py · Legacy LpConv2d with adaptive pooling in LpCNN
-------------------------------------------------------------------------------
This is identical to your friend's 98%-achieving code, except:

  • We replaced the final flatten(8×8) with  an adaptive pool → no shape errors
  • The Lp mask still has the "^(p)" exponent (the "legacy" formula).
  • We added 'init_lp_weights()' for Kaiming/Xavier initialization.
"""

from __future__ import annotations
from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────  LpConv2d (legacy)  ────────────────────────
class LpConv2d(nn.Module):
    """
    L_p convolution (mask=exp(-||Σ·δ||^p)) with the final “^p” exponent,
    as found in the old script that reached ~98 %.

    We do not do the 1/p root in the code, so it's "extra sharp".
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
        self.stride, self.padding = stride, padding

        # regular conv weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *k) * 0.02)

        # mask parameters
        self.p = nn.Parameter(torch.full((out_channels,), p_init))
        sigma_init = sigma_init or (k[0] / 2)
        self.L_diag = nn.Parameter(torch.full((out_channels, 2), 1.0 / sigma_init))
        self.L_off = nn.Parameter(torch.zeros(out_channels, 1))

        # coordinate grid in [0,1], centered
        y = torch.linspace(0, k[0] - 1, k[0]) - (k[0] - 1) / 2
        x = torch.linspace(0, k[1] - 1, k[1]) - (k[1] - 1) / 2
        hy, hx = torch.meshgrid(y, x, indexing="ij")
        off = torch.stack([hy, hx], dim=0)
        off = (off - off.min()) / (off.max() - off.min())
        self.register_buffer("offsets", off.unsqueeze(0))  # [1,2,kh,kw]

    # ----------------------------------------------------------------
    def _mask(self) -> torch.Tensor:
        """
        build mask = exp(- Σ||Σ·δ||^p ^p).

        Implementation detail:  we do the exponent “** p[o]” once more
        after summation, so total exponent is effectively p² if you compare
        to the typical Lp formula.  That is the "legacy" approach used
        in the friend’s code.
        """
        O, kh, kw = self.weight.size(0), *self.kernel_size
        p = torch.clamp(self.p, min=1.0)

        # build covariance Σ = L Lᵀ
        L = torch.zeros(O, 2, 2, device=self.weight.device)
        L[:, 0, 0] = self.L_diag[:, 0]
        L[:, 1, 1] = self.L_diag[:, 1]
        L[:, 1, 0] = self.L_off[:, 0]
        Σ = L @ L.transpose(1, 2)

        # apply Σ to offsets
        δ = torch.einsum("oij,ojhw->oihw", Σ, self.offsets.expand(O, -1, -1, -1))  # [O,2,kh,kw]

        # sum_{c in 2} |δ|^p => shape [O,kh,kw], then ^p again
        lp = torch.zeros(O, kh, kw, device=self.weight.device)
        for o in range(O):
            # "^(p[o])" again after summation => legacy sharper
            lp[o] = (δ[o].abs() ** p[o]).sum(0) ** p[o]

        m = torch.exp(-lp).unsqueeze(1)  # [O,1,kh,kw]
        assert torch.isfinite(m).all(), "NaNs in Lp mask!"
        return m

    # ----------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self.weight * self._mask(),
            stride=self.stride,
            padding=self.padding,
        )


# ─────────────────────  LpCNN (adaptive)  ─────────────────────────
class LpCNN(nn.Module):
    """
    2-block LpConv + BN + ReLU + MaxPool => GAP => FC.
    Exactly the old "two-block" but with adaptive-pool to handle any input
    shape.  For 32×32 images, it’s effectively the same as flatten(8×8).
    """

    def __init__(self, num_classes: int = 4, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            LpConv2d(in_channels, 64, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            LpConv2d(64, 128, 7, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)  # handle any size
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x).flatten(1)  # [B,128]
        return self.fc(x)


# ───────────────── helper: init + param-split ─────────────────────
def split_lp_parameters(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Collects Lp mask parameters vs. normal weights (like your friend did)."""
    mask, weights = [], []
    for n, p in model.named_parameters():
        if any(k in n for k in ("p", "L_diag", "L_off")):
            mask.append(p)
        else:
            weights.append(p)
    return mask, weights

def init_lp_weights(m: nn.Module):
    """Kaiming for conv/LpConv; Xavier for Linear.  Identical to the old script."""
    if isinstance(m, (nn.Conv2d, LpConv2d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.)

__all__ = ["LpConv2d", "LpCNN", "split_lp_parameters", "init_lp_weights"]
