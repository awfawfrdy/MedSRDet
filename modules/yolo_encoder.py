"""Shared YOLOv12n feature encoder — P3/P4/P5 extraction (Appendix B).

This module owns the single encoder implementation used by training, CMCL,
evaluation and inference, so that all code paths observe exactly the same
feature pyramid.

Why a dedicated module
----------------------
The Ultralytics detection model is NOT a plain ``nn.Sequential``: layers carry
``.f`` (from-index) / ``.i`` (layer-index) routing attributes and intermediate
outputs must be kept for skip connections / concatenations.  Iterating layers
and assuming the last maps before the head are P3/P4/P5 is fragile.  This
encoder instead replays the *computation graph* exactly the way
``BaseModel._predict_once`` does, captures the three inputs of the Detect head
(which are, by construction of the head YAML, the P3/P4/P5 pyramid outputs),
and orders them by spatial resolution (P3 = stride 8 = finest).

Channel widths are obtained dynamically from a real forward pass — there is no
hard-coded ``[256, 256, 256]`` silent fallback.  Strides are verified at probe
time (640 x 640 input -> 80/40/20 grid -> strides 8/16/32).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class YOLOv12FeatureEncoder(nn.Module):
    """Replay the Ultralytics computation graph and return P3/P4/P5.

    Args:
        detector: an ``ultralytics`` model object (``YOLO(weight).model``) or
            any ``BaseModel`` exposing ``.model`` as the layer sequence with
            ``.f`` / ``.i`` routing attributes.
        input_size: nominal input resolution used only for the channel/stride
            probe (default 640, matching the manuscript).
    """

    def __init__(self, detector, input_size: int = 640):
        super().__init__()
        m = detector.model if hasattr(detector, "model") else detector
        if not isinstance(m, nn.Sequential):
            # ultralytics DetectionModel wraps the layer sequence in `.model`.
            m = m.model
        if not isinstance(m, nn.Sequential):
            raise TypeError(
                "YOLOv12FeatureEncoder expects an ultralytics model object "
                "(YOLO(weight).model) or its nn.Sequential layer sequence."
            )
        self.model = m
        self.input_size = input_size

        # ---- locate the Detect head and the layers it reads from ----------
        detect_idx, detect_f = None, None
        for i, m in enumerate(self.model):
            if m.__class__.__name__ == "Detect":
                detect_idx, detect_f = i, list(m.f)
                break
        if detect_idx is None:
            raise RuntimeError(
                "YOLOv12FeatureEncoder: no Detect head found in the model; "
                "cannot locate the P3/P4/P5 feature sources."
            )
        self._detect_idx = detect_idx

        # Every layer output referenced by any later layer must be cached.
        save = set()
        for m in self.model:
            f = m.f
            if isinstance(f, int):
                if f != -1:
                    save.add(f)
            elif isinstance(f, (list, tuple)):
                save.update(j for j in f if j != -1)
        self._save = sorted(save)

        # ---- dynamic channel / stride probe with a real forward pass -------
        channels, strides = self._probe(input_size)
        self.channels = channels            # [C_p3, C_p4, C_p5]
        self.strides = strides              # [8, 16, 32] for a 640 input

    # ------------------------------------------------------------------ #
    def _probe(self, input_size: int):
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            feats = self.forward(torch.zeros(1, 3, input_size, input_size))
        if was_training:
            self.model.train()
        channels = [int(f.shape[1]) for f in feats]
        strides = [input_size / int(f.shape[-1]) for f in feats]
        if not (strides[0] == 8 and strides[1] == 16 and strides[2] == 32):
            raise RuntimeError(
                f"unexpected YOLOv12n feature strides {strides}; "
                "expected [8, 16, 32] (P3/P4/P5)."
            )
        return channels, strides

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> list:
        """Return ``[P3, P4, P5]`` feature maps (stride 8 / 16 / 32)."""
        y = []                                    # cached layer outputs
        feats = None
        for i, m in enumerate(self.model):
            if m.f != -1:                          # route inputs like ultralytics
                x = (y[m.f] if isinstance(m.f, int)
                     else [x if j == -1 else y[j] for j in m.f])
            if i == self._detect_idx:
                # The Detect head consumes exactly the P3/P4/P5 maps.
                feats = list(x) if isinstance(x, (list, tuple)) else [x]
                break
            x = m(x)
            y.append(x if i in self._save else None)

        if feats is None:
            raise RuntimeError("Detect head was never reached — broken graph.")

        # Order finest -> coarsest so the contract is always P3, P4, P5
        # regardless of the YAML source ordering.
        feats.sort(key=lambda f: f.shape[-1] * f.shape[-2], reverse=True)
        return feats
