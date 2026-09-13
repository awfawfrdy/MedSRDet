"""Evaluation of MedSRDet: dataset-level class-aware mAP@0.5.

Model selection is performed on the validation set only; the held-out test set
is evaluated once, after the checkpoint has been selected (Section 4.3).

The inference pathway is MFASR -> shared YOLOv12n encoder -> MedHead.  All
CMCL-specific components (projection head, momentum encoder, memory queue) are
absent, exactly as described in Section 3.5 / Methods 3.1.

Key guarantees
--------------
* Dataset routing: the classification head is selected by the ``dataset``
  argument passed to the model; LUNA16 / VinDr-CXR are never evaluated with
  the BraTS head.
* Multi-scale: P3/P4/P5 predictions are decoded with the SAME shared decoder
  as training (``modules.detection_utils.decode_level_boxes``), merged across
  the three levels, and filtered by class-aware NMS.
* Metric: ``compute_map50`` is the standard dataset-level, class-aware AP@0.5
  (predictions collected per class over the whole split; one GT matched at
  most once; COCO 101-point interpolation); VinDr-CXR abnormalities are never
  merged into a single lesion class.

Usage
-----
    python scripts/evaluate.py --config configs/default.yaml \
        --data-root /path/to/prepared --split test \
        --dataset brats2021 --checkpoint runs/exp/seed42/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from modules.detection_utils import compute_map50, postprocess_detections
from scripts.inference import MedSRDetInference, load_model, slice_dataset
from scripts.train import collate


@torch.no_grad()
def run(model, loader: DataLoader, device: str, dataset: str,
        hr_size: int = 640, conf_thres: float = 0.001,
        iou_thres: float = 0.6, max_det: int = 300):
    """Collect per-image predictions and ground truth for one split.

    Args:
        model: MedSRDet / MedSRDetInference; called as ``model(lr,
            dataset=dataset)`` so the dataset-specific classification head is
            used.
        dataset: name of the dataset being evaluated ("brats2021", "luna16",
            "vindrcxr") — REQUIRED, never defaulted silently.

    Returns:
        predictions: {image_id: {"boxes", "scores", "labels"}}
        targets:     {image_id: {"boxes", "labels"}}
    """
    model.eval()
    predictions, targets = {}, {}

    img_id = 0
    for data in loader:
        lr = data["lr"].to(device)
        det = model(lr, dataset=dataset)

        # Multi-scale: decode + merge P3/P4/P5, class-aware NMS.
        results = postprocess_detections(
            det, hr_size=hr_size, conf_thres=conf_thres,
            iou_thres=iou_thres, max_det=max_det,
        )

        for bi, res in enumerate(results):
            key = img_id
            img_id += 1
            predictions[key] = res

            sel = data["batch_idx"] == bi
            gt = data["boxes"][sel].cpu().numpy()
            gt_cls = data["classes"][sel].cpu().numpy().astype(np.int64)
            if len(gt):
                # YOLO cx cy w h (normalised) -> xyxy pixels
                cx, cy, bw, bh = gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3]
                gt_xyxy = np.stack([
                    (cx - bw / 2) * hr_size, (cy - bh / 2) * hr_size,
                    (cx + bw / 2) * hr_size, (cy + bh / 2) * hr_size,
                ], axis=1).astype(np.float32)
            else:
                gt_xyxy = np.zeros((0, 4), dtype=np.float32)
            targets[key] = {"boxes": gt_xyxy, "labels": gt_cls}

    return predictions, targets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dataset", required=True,
                    choices=["brats2021", "luna16", "vindrcxr"])
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    hr_size = cfg["common"]["hr_size"]

    model = load_model(args.checkpoint, cfg, args.device)
    ds = slice_dataset(args.data_root, args.dataset, args.split,
                       hr_size=hr_size, lr_size=cfg["common"]["lr_size"])
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate,
                        shuffle=False)

    preds, targets = run(model, loader, args.device, dataset=args.dataset,
                         hr_size=hr_size)
    map50, precision, recall = compute_map50(preds, targets)

    report = {
        "dataset": args.dataset,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "mAP@0.5": round(map50, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
    }
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
