"""MedSRDet inference pathway.

Section 3.5 / Methods 3.1: CMCL is an auxiliary *training-time* regularizer.
Its projection head, momentum encoder and memory queue are removed from the
deployment graph, so inference consists of

    LR -> MFASR (x4) -> shared YOLOv12n encoder -> MedHead -> boxes + scores

Predictions are decoded with the SAME shared decoder used by training and
evaluation (``modules.detection_utils``), merged across P3/P4/P5 and filtered
by class-aware NMS.  The dataset-specific classification head is selected by
``--dataset``.

Usage
-----
    python scripts/inference.py --config configs/default.yaml \
        --checkpoint runs/exp/seed42/best.pt --input /path/to/lr_images \
        --dataset brats2021 --output detections.json

Output format::

    {
      "<image>.png": [
        {"class_id": 0, "score": 0.87, "bbox_xyxy": [x1, y1, x2, y2]},
        ...
      ]
    }
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
from modules.detection_utils import postprocess_detections
from modules.yolo_encoder import YOLOv12FeatureEncoder


class MedSRDetInference(nn.Module):
    """Deployment-time MedSRDet (no CMCL components)."""

    def __init__(self, cfg: dict, dataset_nc: dict, encoder=None):
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

        if encoder is None:
            from ultralytics import YOLO

            detector = YOLO(m["backbone_weight"]).model
            self.shared_encoder = YOLOv12FeatureEncoder(detector)
        else:
            self.shared_encoder = encoder

        self.encoder_channels = list(self.shared_encoder.channels)

        self.medhead = MedHead(
            in_channels=self.encoder_channels,      # P3, P4, P5
            channels=m["unified_channels"],
            dataset_nc=dataset_nc,
        )

    def encode(self, x):
        """P3/P4/P5 feature maps from the shared encoder."""
        return self.shared_encoder(x)

    def forward(self, lr: torch.Tensor, dataset: str = "brats2021"):
        sr = self.mfasr(lr)
        p345 = self.encode(sr)
        return self.medhead(p345, dataset=dataset)


def load_model(checkpoint: str, cfg: dict, device: str,
               encoder=None) -> MedSRDetInference:
    dataset_nc = {d: cfg["datasets"][d]["num_classes"]
                  for d in ("brats2021", "luna16", "vindrcxr")}

    model = MedSRDetInference(cfg, dataset_nc, encoder=encoder).to(device)

    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model", ckpt)

    # Training checkpoints also carry CMCL buffers (queue, queue_ptr, momentum
    # encoder).  They are not part of the inference graph and are dropped
    # silently.
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()

    if missing:
        print(f"[warn] {len(missing)} parameters not restored from checkpoint")
    return model


def slice_dataset(data_root, dataset: str, split: str,
                  hr_size: int = 640, lr_size: int = 160, seed: int = 42):
    """Rebuild a prepared split from ``scripts/preprocess.py`` output.

    Validation/test splits keep every eligible slice (no 1:1 negative
    balancing), matching the evaluation protocol.
    """
    from scripts.train import SliceDataset

    return SliceDataset(Path(data_root) / dataset, split, dataset_name=dataset,
                        hr_size=hr_size, lr_size=lr_size, seed=seed,
                        balance_negatives=False)


def predict_images(model, image_dir: Path, device: str, dataset: str,
                   hr_size: int = 640, conf_thres: float = 0.25,
                   iou_thres: float = 0.6, max_det: int = 300) -> dict:
    """Run detection on every PNG in ``image_dir``; returns JSON-ready dict."""
    import cv2

    results = {}
    for path in sorted(Path(image_dir).glob("*.png")):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)
        tensor = torch.from_numpy(np.ascontiguousarray(img)) \
                      .permute(2, 0, 1).unsqueeze(0).float().to(device)

        with torch.no_grad():
            out = model(tensor, dataset=dataset)

        dets = postprocess_detections(out, hr_size=hr_size,
                                      conf_thres=conf_thres,
                                      iou_thres=iou_thres, max_det=max_det)[0]
        results[path.name] = [
            {
                "class_id": int(lab),
                "score": float(score),
                "bbox_xyxy": [round(float(v), 2) for v in box],
            }
            for box, score, lab in zip(dets["boxes"], dets["scores"],
                                       dets["labels"])
        ]
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True, help="directory of LR images")
    ap.add_argument("--dataset", default="brats2021",
                    choices=["brats2021", "luna16", "vindrcxr"])
    ap.add_argument("--output", default="detections.json")
    ap.add_argument("--conf-thres", type=float, default=0.25)
    ap.add_argument("--iou-thres", type=float, default=0.6)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    model = load_model(args.checkpoint, cfg, args.device)

    results = predict_images(
        model, Path(args.input), args.device, args.dataset,
        hr_size=cfg["common"]["hr_size"], conf_thres=args.conf_thres,
        iou_thres=args.iou_thres, max_det=args.max_det,
    )

    Path(args.output).write_text(json.dumps(results, indent=2))
    n_det = sum(len(v) for v in results.values())
    print(f"wrote {len(results)} images / {n_det} detections to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
