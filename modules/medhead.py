"""MedHead — small-lesion detection head (aligned implementation).

Follows the revised manuscript (Methods 3.4 / Tables 2-4).

Manuscript-explicit design:
    - Input: multi-scale features from the backbone feature pyramid.
    - Shared projection stage: 1x1 Conv projects every scale to 256 channels.
    - Three PARALLEL attention branches (NOT serial):
        A. Scale-aware   : GAP -> MLP 256->64->256 (ReLU) -> Sigmoid modulation.
        B. Spatial-aware : deformable convolution, kernel 3x3, groups = 8,
                           with offset prediction for the 3x3 deformable conv.
        C. Task-aware    : Dynamic ReLU / task-adaptive gating with dynamic
                           alpha1, alpha2, beta1, beta2.
    - Branch contextual processing: lightweight 3x3 conv per branch.
    - Fusion: channel concatenation -> 1x1 Conv -> 256 -> SiLU.
    - Residual connection: out = x_proj + gamma * fused, gamma = 0.2.
    - Detection outputs: dataset-specific 1x1 classification head
      (Conv2d(256, N_cls^(d), 1)) and a 1x1 regression head
      (Conv2d(256, 4, 1)) predicting dx, dy, dw, dh.

Manuscript-unspecified implementation details (reported in the audit):
    - Appendix B (Table B.1): "the three feature maps are INDEPENDENTLY
      projected to a unified channel dimension of 256 through 1x1
      convolutional layers".  Accordingly the default builds one dedicated
      1x1 projection per scale (P3/P4/P5) instead of a single shared one.
    - The three parallel attention branches and the fusion layers are shared
      across scales (standard FPN-style head weight sharing).
    - Dynamic-ReLU coefficient generation network uses a hidden width of 64
      (no hidden ratio is specified in the manuscript).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.ops import DeformConv2d as _TVDeformConv2d
    _HAS_DCN = True
except Exception:  # pragma: no cover - degraded environment
    _TVDeformConv2d = None
    _HAS_DCN = False

# Manuscript constants (Methods 3.4)
FEATURE_CHANNELS_DEFAULT = 256
SCALE_HIDDEN_DEFAULT = 64
DEFORM_KERNEL_DEFAULT = 3
DEFORM_GROUPS_DEFAULT = 8
RESIDUAL_GAMMA_DEFAULT = 0.2
REGRESSION_OUT_DEFAULT = 4  # dx, dy, dw, dh


class ScaleAwareAttention(nn.Module):
    """Scale-aware (channel) attention: GAP -> MLP 256->64->256 -> Sigmoid."""

    def __init__(self, channels: int = FEATURE_CHANNELS_DEFAULT,
                 hidden: int = SCALE_HIDDEN_DEFAULT):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.gap(x).flatten(1)          # [B, C]
        w = self.sigmoid(self.mlp(w))       # [B, C]
        return x * w.view(x.shape[0], x.shape[1], 1, 1)


class SpatialAwareAttention(nn.Module):
    """Spatial-aware attention via 3x3 deformable convolution (groups = 8)."""

    def __init__(self, channels: int = FEATURE_CHANNELS_DEFAULT,
                 kernel_size: int = DEFORM_KERNEL_DEFAULT,
                 groups: int = DEFORM_GROUPS_DEFAULT):
        super().__init__()
        if not _HAS_DCN:
            raise ImportError(
                "torchvision.ops.DeformConv2d is required by MedHead's "
                "spatial-aware attention branch."
            )
        self.groups = groups
        self.kernel_size = kernel_size
        pad = kernel_size // 2
        # Offset channels needed by torchvision DeformConv2d:
        #   offset = 2 * offset_groups * kH * kW  (offset_groups == groups).
        self.offset_conv = nn.Conv2d(
            channels, 2 * kernel_size * kernel_size * groups, kernel_size, 1, pad
        )
        self.deform_conv = _TVDeformConv2d(
            channels, channels, kernel_size, stride=1, padding=pad, groups=groups
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offsets = self.offset_conv(x)        # [B, 2*k*k*groups, H, W]
        return self.deform_conv(x, offsets)


class TaskAwareAttention(nn.Module):
    """Task-aware attention via Dynamic ReLU (dynamic alpha1/alpha2/beta1/beta2).

    y = max(alpha1(x) * x + beta1(x), alpha2(x) * x + beta2(x))

    Coefficients are generated from the global context of the input feature by
    a lightweight MLP (hidden width 64; not specified in the manuscript).
    """

    def __init__(self, channels: int = FEATURE_CHANNELS_DEFAULT,
                 hidden: int = SCALE_HIDDEN_DEFAULT):
        super().__init__()
        self.channels = channels
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.coef_net = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 4 * channels),  # alpha1, alpha2, beta1, beta2
        )
        self._init_coef_net()

    def _init_coef_net(self) -> None:
        # ReLU-initialized Dynamic ReLU: the last coefficient Linear is
        # zero-initialized (weight = bias = 0).  Under the parameterization
        #     alpha1 = 1 + tanh(raw_alpha1)
        #     alpha2 =     tanh(raw_alpha2)
        #     beta1  =     tanh(raw_beta1)
        #     beta2  =     tanh(raw_beta2)
        # a zero coefficient output gives exactly alpha1=1, alpha2=0,
        # beta1=beta2=0, i.e. y = max(1*x + 0, 0*x + 0) = max(x, 0) = ReLU(x).
        last = self.coef_net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def compute_coefficients(self, x: torch.Tensor):
        """Return dynamic (alpha1, alpha2, beta1, beta2) shaped [B, C, 1, 1]."""
        B, C = x.shape[0], x.shape[1]
        ctx = self.gap(x).flatten(1)               # [B, C]
        coef = self.coef_net(ctx)                  # [B, 4*C]
        raw_a1, raw_a2, raw_b1, raw_b2 = torch.split(coef, C, dim=1)
        # ReLU-initialized DY-ReLU parameterization (see _init_coef_net):
        # at initialization coef == 0 => alpha1=1, alpha2=0, beta1=beta2=0.
        # After training the four coefficient groups remain dynamic functions
        # of the input global context.
        a1 = (1.0 + torch.tanh(raw_a1)).view(B, C, 1, 1)
        a2 = torch.tanh(raw_a2).view(B, C, 1, 1)
        b1 = torch.tanh(raw_b1).view(B, C, 1, 1)
        b2 = torch.tanh(raw_b2).view(B, C, 1, 1)
        return a1, a2, b1, b2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a1, a2, b1, b2 = self.compute_coefficients(x)
        return torch.maximum(a1 * x + b1, a2 * x + b2)


class MedHead(nn.Module):
    """Small-lesion detection head with parallel multi-attention branches.

    Args:
        in_channels: int, or list[int] aligned with the pyramid scales.  Each
            scale is projected to 256 channels by its OWN 1x1 convolution
            (Appendix B, Table B.1), which is the default
            (``in_channels=(256, 256, 256)``).  A single int is still accepted
            for backwards compatibility and then falls back to one shared
            1x1 projection across scales.
        nc: int, number of classes (used for the default classification head).
            For multiple datasets pass ``dataset_nc`` instead.
        dataset_nc: dict[str, int] mapping dataset id -> number of classes.
            Builds dataset-specific classification heads (ModuleDict).
        channels / hidden / kernel_size / groups / gamma: manuscript constants.
    """

    def __init__(
        self,
        in_channels=(FEATURE_CHANNELS_DEFAULT,) * 3,   # P3, P4, P5
        nc: int = 1,
        dataset_nc: dict | None = None,
        channels: int = FEATURE_CHANNELS_DEFAULT,
        hidden: int = SCALE_HIDDEN_DEFAULT,
        kernel_size: int = DEFORM_KERNEL_DEFAULT,
        groups: int = DEFORM_GROUPS_DEFAULT,
        gamma: float = RESIDUAL_GAMMA_DEFAULT,
    ):
        super().__init__()
        self.channels = channels
        self.gamma = gamma
        self.in_channels = in_channels

        # ---- Per-scale projection stage: P3/P4/P5 -> 1x1 conv -> 256 ----
        # Appendix B (Table B.1): the three feature maps are INDEPENDENTLY
        # projected to 256 channels, i.e. one 1x1 convolution per scale.
        if isinstance(in_channels, int):
            self.projection = nn.Conv2d(in_channels, channels, 1)
            self._per_scale_proj = False
        else:
            self.projection = nn.ModuleList(
                [nn.Conv2d(ci, channels, 1) for ci in in_channels]
            )
            self._per_scale_proj = True

        # ---- Three PARALLEL attention branches ----
        self.scale_attn = ScaleAwareAttention(channels, hidden)
        self.spatial_attn = SpatialAwareAttention(channels, kernel_size, groups)
        self.task_attn = TaskAwareAttention(channels, hidden)

        # ---- Branch contextual processing (lightweight 3x3 conv) ----
        self.ctx_conv = nn.ModuleList(
            [nn.Conv2d(channels, channels, 3, 1, 1) for _ in range(3)]
        )

        # ---- Fusion: concat -> 1x1 conv -> 256 -> SiLU ----
        self.fusion_conv = nn.Conv2d(channels * 3, channels, 1)

        # ---- Regression head (shared): Conv2d(256, 4, 1) -> dx,dy,dw,dh ----
        self.reg_head = nn.Conv2d(channels, REGRESSION_OUT_DEFAULT, 1)

        # ---- Dataset-specific classification heads ----
        if dataset_nc is not None:
            self.dataset_nc = dict(dataset_nc)
            self.cls_heads = nn.ModuleDict(
                {name: nn.Conv2d(channels, n, 1) for name, n in self.dataset_nc.items()}
            )
            self.cls_head = None
        else:
            self.dataset_nc = {"default": int(nc)}
            self.cls_heads = nn.ModuleDict({"default": nn.Conv2d(channels, int(nc), 1)})
            self.cls_head = self.cls_heads["default"]

    # ------------------------------------------------------------------ #
    def _project(self, x: torch.Tensor, scale_idx: int) -> torch.Tensor:
        if self._per_scale_proj:
            return self.projection[scale_idx](x)
        return self.projection(x)

    def feature_refine(self, x: torch.Tensor | list[torch.Tensor]
                       ) -> torch.Tensor | list[torch.Tensor]:
        """Run the shared projection + parallel attention + fusion + residual.

        Output channels are 256 for every scale.
        """
        single = isinstance(x, torch.Tensor)
        xs = [x] if single else list(x)

        outs = []
        for i, xi in enumerate(xs):
            proj = self._project(xi, i)               # [B, 256, H, W]
            s = self.scale_attn(proj)                 # parallel branch A
            sp = self.spatial_attn(proj)              # parallel branch B
            t = self.task_attn(proj)                  # parallel branch C
            s = self.ctx_conv[0](s)
            sp = self.ctx_conv[1](sp)
            t = self.ctx_conv[2](t)
            fused = self.fusion_conv(torch.cat([s, sp, t], dim=1))  # concat -> 1x1
            fused = F.silu(fused)
            outs.append(proj + self.gamma * fused)    # weighted residual, gamma=0.2

        return outs[0] if single else outs

    def classify(self, feat: torch.Tensor | list[torch.Tensor],
                 dataset: str = "default"
                 ) -> torch.Tensor | list[torch.Tensor]:
        head = self.cls_heads[dataset]
        if isinstance(feat, torch.Tensor):
            return head(feat)
        return [head(f) for f in feat]

    def regress(self, feat: torch.Tensor | list[torch.Tensor]
                ) -> torch.Tensor | list[torch.Tensor]:
        if isinstance(feat, torch.Tensor):
            return self.reg_head(feat)
        return [self.reg_head(f) for f in feat]

    def forward(self, x: torch.Tensor | list[torch.Tensor],
                dataset: str = "default"
                ) -> dict:
        """Return {'features', 'cls', 'bbox'} with the same structure as x.

        For a tensor input every output is a tensor; for a list of scales every
        output is a list of per-scale tensors.
        """
        feats = self.feature_refine(x)
        return {
            "features": feats,
            "cls": self.classify(feats, dataset),
            "bbox": self.regress(feats),
        }
