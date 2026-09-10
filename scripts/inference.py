"""MedSRDet inference pathway.

Section 3.5 / Methods 3.1: CMCL is an auxiliary *training-time* regularizer.
Its projection head, momentum encoder and memory queue are removed from the
deployment graph, so inference consists of

    LR -> MFASR (x4) -> shared YOLOv12n encoder -> MedHead -> boxes + scores

Usage
-----
    python scripts/inference.py --config configs/default.yaml \
        --checkpoint runs/exp/seed42/best.pt --input /path/to/lr_images \
        --output detections.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from modules import MFASRNet, MedHead


class MedSRDetInference(nn.Module):
    """Deployment-time MedSRDet (no CMCL components)."""

    def __init__(self, cfg: dict, dataset_nc: dict):
        super().__init__()

        m = cfg["model"]

        self.mfasr = MFASRNet(
            num_in_ch=m["mfasr"]["in_channels"],
            num_out_ch=m["mfasr"]["out_channels"],
            num_feat=m["mfasr"]["num_feat"],
            num_block=m["mfasr"]["num_block"],
            num_grow_ch=m["mfasr"]["growth_channels"],
            scale=m["mfasr"]["scale"],
        )

        from ultralytics import YOLO

        self.detector = YOLO(m["backbone_weight"]).model
        self.encoder_channels = self._probe_encoder_channels()

        self.medhead = MedHead(
            in_channels=self.encoder_channels,      # P3, P4, P5
            channels=m["unified_channels"],
            dataset_nc=dataset_nc,
        )

    def _probe_encoder_channels(self):
        for mod in self.detector.modules():
            if mod.__class__.__name__ == "Detect":
                try:
                    return [int(c) for c in mod.cv2[0][0].in_channels][:3]
                except Exception:
                    pass
        return [256, 256, 256]

    def encode(self, x):
        """P3/P4/P5 feature maps from the shared encoder."""
        feats = []
        for layer in self.detector.model:
            x = layer(x)
            feats.append(x)
        return feats[-4:-1]

    def forward(self, lr: torch.Tensor, dataset: str = "brats2021"):
        sr = self.mfasr(lr)
        p345 = self.encode(sr)
        out = self.medhead(p345, dataset=dataset)
        return out


def load_model(checkpoint: str, cfg: dict, device: str) -> MedSRDetInference:
    dataset_nc = {d: cfg["datasets"][d]["num_classes"]
                  for d in ("brats2021", "luna16", "vindrcxr")}

    model = MedSRDetInference(cfg, dataset_nc).to(device)

    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model", ckpt)

    # Training checkpoints also carry CMCL buffers (queue, queue_ptr).  They
    # are not part of the inference graph and are dropped silently.
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()

    if missing:
        print(f"[warn] {len(missing)} parameters not restored from checkpoint")
    return model


def slice_dataset(data_root, dataset: str, split: str,
                  hr_size: int = 640, lr_size: int = 160, seed: int = 42):
    """Rebuild a prepared split from ``scripts/preprocess.py`` output."""
    from scripts.train import SliceDataset

    return SliceDataset(Path(data_root) / dataset, split,
                        hr_size=hr_size, lr_size=lr_size, seed=seed)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True, help="directory of LR images")
    ap.add_argument("--dataset", default="brats2021")
    ap.add_argument("--output", default="detections.json")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    model = load_model(args.checkpoint, cfg, args.device)

    import cv2

    results = {}
    for path in sorted(Path(args.input).glob("*.png")):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)
        tensor = torch.from_numpy(np.ascontiguousarray(img)) \
                      .permute(2, 0, 1).unsqueeze(0).float().to(args.device)

        with torch.no_grad():
            out = model(tensor, dataset=args.dataset)

        results[path.name] = {
            "cls": [c.shape for c in out["cls"]],
            "bbox": [b.shape for b in out["bbox"]],
        }

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"wrote {len(results)} entries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
