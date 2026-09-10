"""CMCL — Cross-view Modality-robust Contrastive Learning (aligned implementation).

Follows the revised manuscript (Methods 3.3 / Tables 2-4).

Key semantics (manuscript-explicit):
    - A *positive pair* is formed by two views of the SAME annotated lesion
      instance:  (augmented view_i, MFASR-enhanced view_i).
      CMCL does NOT build explicit cross-modal positives such as
      CT <-> MRI, CT <-> X-ray, or MRI <-> X-ray, and does not require a
      unified disease taxonomy across datasets.
    - The query representation comes from the shared detection backbone
      features; this module does not own an independent detection backbone.
    - Modality only implies heterogeneous-domain exposure in a mixed-modal
      mini-batch; identical/different modality never makes two instances a
      positive pair automatically.  Only the same-lesion view pair is positive;
      every other lesion instance (whatever its modality) is a negative.
    - Training-only: the projection head, momentum encoder, and memory queue
      are used during training.  Inference of MedSRDet does not depend on any
      CMCL-specific component.

Manuscript-specified multi-scale aggregation (Appendix B, Table B.1):
    Each lesion-centered view is encoded by the shared YOLOv12n encoder.  The
    P3, P4 and P5 representations are independently projected to 256 channels
    (1x1 convolution), spatially aggregated by GLOBAL AVERAGE POOLING, and the
    resulting three 256-dimensional vectors are fused by ELEMENT-WISE MEAN into
    a single 256-dimensional lesion representation, which is then mapped to the
    128-dimensional contrastive embedding.

Manuscript-unspecified implementation details (documented for reviewers):
    - Standard InfoNCE formulation with in-batch + queue negatives.
    - Frobenius regularization on projection-head linear weights.
    - Momentum encoder (Table 3): the projection head is mirrored as g_phi_m
      by EMA, which is what the memory queue keys are produced with.  The
      shared YOLOv12n feature encoder can additionally be mirrored as
      E_theta_m through ``attach_momentum_backbone()``; that EMA copy encodes
      the KEY view only and never replaces the online backbone used for
      detection, so the parameter sharing of Methods 3.1 is preserved.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Manuscript constants (Methods 3.3)
FEATURE_DIM_DEFAULT = 256
EMBEDDING_DIM_DEFAULT = 128
PROJ_HIDDEN_DEFAULT = 256
DROPOUT_DEFAULT = 0.2
TEMPERATURE_DEFAULT = 0.07
LAMBDA_REG_DEFAULT = 1e-4
MOMENTUM_DEFAULT = 0.999
QUEUE_SIZE_DEFAULT = 65536


class _MLPProjector(nn.Module):
    """2-layer MLP projection head used by both query and (EMA) key encoders.

    Architecture (manuscript-explicit): 2-layer MLP, ReLU, Dropout=0.2,
    mapping 256 -> 128 contrastive embedding followed by L2 normalization
    (performed by the caller).
    """

    def __init__(self, in_dim: int = FEATURE_DIM_DEFAULT,
                 hidden_dim: int = PROJ_HIDDEN_DEFAULT,
                 out_dim: int = EMBEDDING_DIM_DEFAULT,
                 dropout: float = DROPOUT_DEFAULT):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x


class CMCL(nn.Module):
    """Cross-view contrastive regularizer (training only).

    Usage contract (backbone/shared-feature integration):
        aug_feat = shared_backbone(augmented_lesion_view)     # [B, C, H, W] or [B, C]
        sr_feat  = shared_backbone(mfasr_enhanced_lesion_view)  # [B, C, H, W] or [B, C]
        cmcl_loss = cmcl(aug_feat, sr_feat)  # positive pair = same index i

    The module never crops lesion patches (the data pipeline owns that) and
    never creates an independent feature encoder.
    """

    def __init__(
        self,
        in_channels=None,
        feature_dim: int = FEATURE_DIM_DEFAULT,
        embedding_dim: int = EMBEDDING_DIM_DEFAULT,
        hidden_dim: int = PROJ_HIDDEN_DEFAULT,
        dropout: float = DROPOUT_DEFAULT,
        temperature: float = TEMPERATURE_DEFAULT,
        lambda_reg: float = LAMBDA_REG_DEFAULT,
        momentum: float = MOMENTUM_DEFAULT,
        queue_size: int = QUEUE_SIZE_DEFAULT,
    ):
        super().__init__()
        if embedding_dim != EMBEDDING_DIM_DEFAULT:
            raise ValueError(
                "manuscript fixes the contrastive embedding dimension to 128; "
                f"got embedding_dim={embedding_dim}"
            )
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.lambda_reg = lambda_reg
        self.momentum = momentum
        self.queue_size = queue_size

        # E_theta_m: EMA copy of the shared feature encoder (Table 3).
        # Optional - created on demand by attach_momentum_backbone().
        self.momentum_backbone = None

        # ---- Per-scale 1x1 projections: P3/P4/P5 -> feature_dim (256) ----
        # Appendix B (Table B.1): "the P3, P4, and P5 representations are
        # independently projected to 256 channels" -> one dedicated 1x1
        # convolution per scale, i.e. weights are NOT shared across scales.
        self._per_scale_proj = isinstance(in_channels, (list, tuple))
        if in_channels is None:
            # Unknown backbone channel widths: projection is skipped and the
            # caller is responsible for feeding feature_dim-channel maps.
            self.scale_proj = None
        elif self._per_scale_proj:
            self.scale_proj = nn.ModuleList(
                [nn.Conv2d(int(ci), feature_dim, 1) for ci in in_channels]
            )
        else:
            self.scale_proj = (
                None if int(in_channels) == feature_dim
                else nn.Conv2d(int(in_channels), feature_dim, 1)
            )

        # Query encoder (trainable projection head).
        self.query_proj = _MLPProjector(feature_dim, hidden_dim, embedding_dim, dropout)

        # Key / momentum encoder: EMA copy of the projection head.  It is NOT a
        # separate backbone; it shares the same architecture and is updated as
        # theta_k = m * theta_k + (1 - m) * theta_q.  It never receives grads.
        self.key_encoder = _MLPProjector(feature_dim, hidden_dim, embedding_dim, dropout=0.0)
        for p in self.key_encoder.parameters():
            p.requires_grad = False
        self._init_ema_params()

        # FIFO memory queue of historical lesion keys (L2 normalized), K = 65536.
        # Shape [K, embedding_dim]; queue feature stored as the transposed
        # [embedding_dim, K] is NOT used: paper only fixes K and the embedding
        # dim, layout [K, dim] is the canonical MoCo convention.
        queue = F.normalize(torch.randn(queue_size, embedding_dim), dim=1)
        self.register_buffer("queue", queue)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    # ------------------------------------------------------------------ #
    # EMA helpers
    # ------------------------------------------------------------------ #
    def _init_ema_params(self) -> None:
        with torch.no_grad():
            for p_q, p_k in zip(self.query_proj.parameters(), self.key_encoder.parameters()):
                p_k.data.copy_(p_q.data)

    @torch.no_grad()
    def attach_momentum_backbone(self, backbone: nn.Module) -> None:
        """Create E_theta_m, the EMA copy of the shared feature encoder.

        Table 3 lists the momentum encoder as ``E_theta_m(.), g_phi_m(.)``.
        The copy is frozen and kept in eval mode; it encodes the KEY view only.
        The ONLINE view is always encoded by the real shared backbone, so the
        contrastive objective still regularizes exactly the features used for
        lesion detection (Methods 3.1).
        """
        self.momentum_backbone = copy.deepcopy(backbone)
        for p in self.momentum_backbone.parameters():
            p.requires_grad_(False)
        self.momentum_backbone.eval()

    @torch.no_grad()
    def update_momentum_encoder(self, online_backbone=None) -> None:
        """EMA update of the momentum encoder (Table 3).

        theta_k <- m * theta_k + (1 - m) * theta_q

        Args:
            online_backbone: shared feature encoder E_theta.  When given and
                E_theta_m has been attached, its parameters and buffers are
                mirrored into E_theta_m as well.
        """
        m = self.momentum
        for p_q, p_k in zip(self.query_proj.parameters(), self.key_encoder.parameters()):
            p_k.data = p_k.data * m + p_q.data * (1.0 - m)

        if online_backbone is None or self.momentum_backbone is None:
            return

        for p_q, p_k in zip(online_backbone.parameters(),
                            self.momentum_backbone.parameters()):
            p_k.data.mul_(m).add_(p_q.data, alpha=1.0 - m)

        online_buffers = dict(online_backbone.named_buffers())
        for name, buf_k in self.momentum_backbone.named_buffers():
            if name in online_buffers:
                buf_k.copy_(online_buffers[name])

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        """FIFO circular queue update with the current mini-batch keys.

        Supports an arbitrary number of keys per step (the number of lesion
        instances in a mini-batch is data-dependent and generally not a
        divisor of the queue size): keys are written at the FIFO pointer and,
        if they exceed the remaining capacity, the tail wraps around and
        overwrites the oldest entries (standard circular-buffer behaviour).
        """
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr.item())
        n = keys.shape[0]
        if ptr + n <= self.queue_size:
            self.queue[ptr: ptr + n] = keys
            ptr = (ptr + n) % self.queue_size
        else:
            # Wrap-around write.
            first = self.queue_size - ptr
            self.queue[ptr:] = keys[:first]
            self.queue[: n - first] = keys[first:]
            ptr = (ptr + n) % self.queue_size
        self.queue_ptr[0] = ptr

    # ------------------------------------------------------------------ #
    # Pooling helpers (backbone features are spatial maps)
    # ------------------------------------------------------------------ #
    def _pool_and_project(self, x: torch.Tensor, scale_idx: int) -> torch.Tensor:
        """1x1 projection (per scale) + global average pooling -> [B, feature_dim]."""
        if x.dim() == 4:
            if self.scale_proj is not None:
                proj = (self.scale_proj[scale_idx] if self._per_scale_proj
                        else self.scale_proj)
                x = proj(x)
            elif x.shape[1] != self.feature_dim:
                raise ValueError(
                    "CMCL expects feature maps with "
                    f"{self.feature_dim} channels (Appendix B, Table B.1) when "
                    f"`in_channels` is not given; got {x.shape[1]}. Pass "
                    "`in_channels=(c_p3, c_p4, c_p5)` to build the per-scale "
                    "1x1 projections."
                )
            x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return x

    def _to_vector(self, x) -> torch.Tensor:
        """Aggregate the shared-backbone features of one lesion view -> [B, feature_dim].

        Implements Appendix B (Table B.1): the P3/P4/P5 representations are
        independently projected to 256 channels, spatially aggregated with
        global average pooling, and the three resulting 256-d vectors are fused
        by ELEMENT-WISE MEAN into a single lesion representation.

        Args:
            x: either a single tensor ``[B, C, H, W]`` / ``[B, C]`` (single-scale
               fallback, kept for backwards compatibility), or a list/tuple of
               the P3, P4, P5 feature maps.
        """
        if isinstance(x, (list, tuple)):
            vecs = [self._pool_and_project(xi, i) for i, xi in enumerate(x)]
            return torch.stack(vecs, dim=0).mean(dim=0)   # element-wise mean
        return self._pool_and_project(x, 0)

    def _embed_query(self, x: torch.Tensor) -> torch.Tensor:
        z = self._to_vector(x)
        z = self.query_proj(z)
        return F.normalize(z, dim=1)

    @torch.no_grad()
    def _embed_key(self, x: torch.Tensor) -> torch.Tensor:
        z = self._to_vector(x)
        self.key_encoder.eval()  # keep dropout off on the EMA branch
        z = self.key_encoder(z)
        return F.normalize(z, dim=1)

    # ------------------------------------------------------------------ #
    # Regularization: lambda_reg * ||W_phi||_F^2 on projection-head weights
    # ------------------------------------------------------------------ #
    def _projection_regularization(self) -> torch.Tensor:
        reg = torch.zeros((), device=self.query_proj.fc1.weight.device)
        for name, p in self.query_proj.named_parameters():
            if "weight" in name and p.dim() >= 2:  # W_phi matrices only
                reg = reg + p.pow(2).sum()
        return reg

    # ------------------------------------------------------------------ #
    # Main
    # ------------------------------------------------------------------ #
    def forward(self, aug_feat: torch.Tensor, sr_feat: torch.Tensor) -> torch.Tensor:
        """Contrastive loss for one mini-batch of lesion instances.

        Args:
            aug_feat: shared-backbone feature of the AUGMENTED view, index i.
            sr_feat:  shared-backbone feature of the MFASR-ENHANCED view of the
                      SAME lesion index i.
        Both follow the manuscript interface (Appendix B, Table B.1): a list of
        the P3/P4/P5 feature maps ``([B,C3,H3,W3], [B,C4,H4,W4], [B,C5,H5,W5])``
        of the corresponding 224x224 lesion view.  A single ``[B, C, H, W]`` or
        ``[B, C]`` tensor is still accepted as a single-scale fallback.
        Positives are (aug_i, sr_i); negatives are all other in-batch lesion
        keys and the memory queue.

        Returns:
            A zero scalar in eval mode (inference must not depend on CMCL),
            otherwise InfoNCE loss + lambda_reg * ||W_phi||_F^2.
        """
        if not self.training:
            # Inference must not depend on CMCL (Methods 3.5).  `aug_feat` may
            # be a list of P3/P4/P5 maps, so resolve the device from the first
            # element in that case.
            ref = aug_feat[0] if isinstance(aug_feat, (list, tuple)) else aug_feat
            return torch.zeros((), device=ref.device)

        q = self._embed_query(aug_feat)                      # [B, d]
        k = self._embed_key(sr_feat)                         # [B, d]

        # logits layout: first B columns are sim(q_i, k_j) (j=i is the positive
        # same-lesion pair), remaining K columns are sim(q_i, queue_m).
        # The queue must NOT be part of the autograd graph of the current step
        # (it stores historical no-grad keys and is updated in-place right after
        # computing logits).  detach()+clone() breaks storage sharing so the
        # in-place enqueue below cannot invalidate the saved tensor of the
        # matmul during backward (standard MoCo-style trick).
        queue = self.queue.detach().clone()
        logits = torch.cat([q @ k.T, q @ queue.T], dim=1) / self.temperature  # [B, B+K]
        labels = torch.arange(logits.shape[0], device=logits.device, dtype=torch.long)
        info_nce = F.cross_entropy(logits, labels)

        # Projection-head regularization (explicit formula, not weight decay).
        reg = self.lambda_reg * self._projection_regularization()

        # Update the FIFO memory queue with the current keys.
        self._dequeue_and_enqueue(k.detach())

        return info_nce + reg

    # ------------------------------------------------------------------ #
    # Full Table-3 momentum path: E_theta_m + g_phi_m
    # ------------------------------------------------------------------ #
    def forward_from_images(self, aug_images, sr_images, encoder) -> torch.Tensor:
        """InfoNCE computed from raw lesion views through the FULL momentum
        encoder of Table 3:

            q = g_phi(E_theta(x_aug)),     k = g_phi_m(E_theta_m(x_sr)).

        ``encoder`` is the shared YOLOv12n feature encoder E_theta.  On the
        first call it is deep-copied into the frozen momentum copy E_theta_m
        (see ``attach_momentum_backbone``); after each optimiser step the copy
        is refreshed by ``update_momentum_encoder(encoder)``, so the online
        encoder keeps receiving the detection gradients (Methods 3.1).

        The key branch shares the per-scale 1x1 projections (they form the
        interface to the backbone widths) but its projection head is the
        momentum copy g_phi_m, and no gradient flows into the key branch.

        The feature-level ``forward`` above remains available for callers that
        pre-extract backbone features.
        """
        if not self.training:
            ref = aug_images[0] if isinstance(aug_images, (list, tuple)) else aug_images
            return torch.zeros((), device=ref.device)

        if self.momentum_backbone is None:
            self.attach_momentum_backbone(encoder)

        # Query branch: online shared encoder; gradients flow into E_theta.
        q = self._embed_query(encoder(aug_images))

        # Key branch: frozen E_theta_m -> (shared 1x1 + GAP + mean) -> g_phi_m.
        with torch.no_grad():
            k_feats = self.momentum_backbone(sr_images)
            k_vec = self._to_vector(k_feats)
            self.key_encoder.eval()
            k = F.normalize(self.key_encoder(k_vec), dim=1)

        queue = self.queue.detach().clone()
        logits = torch.cat([q @ k.T, q @ queue.T], dim=1) / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device, dtype=torch.long)
        info_nce = F.cross_entropy(logits, labels)
        reg = self.lambda_reg * self._projection_regularization()

        self._dequeue_and_enqueue(k.detach())
        return info_nce + reg
