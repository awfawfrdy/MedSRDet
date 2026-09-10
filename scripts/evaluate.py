"""Evaluation of MedSRDet: mAP@0.5, precision and recall.

Model selection is performed on the validation set only; the held-out test set
is evaluated once, after the checkpoint has been selected (Section 4.3).

The inference pathway is MFASR -> shared YOLOv12n encoder -> MedHead.  All
CMCL-specific components (projection head, momentum encoder, memory queue) are
absent, exactly as described in Section 3.5 / Methods 3.1.

Usage
-----
    python scripts/evaluate.py --config configs/default.yaml \
        --data-root /path/to/prepared --split test \
        --checkpoint runs/exp/seed42/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from scripts.inference import MedSRDetInference, load_model, slice_dataset
from scripts.train import collate


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes in (x1, y1, x2, y2) form."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)

    area_a = np.clip(a[:, 2:] - a[:, :2], 0, None).prod(axis=1)
    area_b = np.clip(b[:, 2:] - b[:, :2], 0, None).prod(axis=1)

    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-8)


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO-style AP: precision envelope integrated over 101 recall points."""
    if len(recall) == 0:
        return 0.0

    order = np.argsort(recall)
    r = recall[order]
    p = precision[order]

    # Monotone decreasing precision envelope.
    p = np.maximum.accumulate(p[::-1])[::-1]

    points = np.linspace(0.0, 1.0, 101)
    return float(np.sum(np.interp(points, r, p)) / 101.0)


def compute_map50(predictions: dict, targets: dict, iou_threshold: float = 0.5):
    """mAP@0.5 over a set of images.

    Args:
        predictions: {image_id: (boxes[N,4], scores[N])} in xyxy pixel coords.
        targets:     {image_id: boxes[M,4]} in xyxy pixel coords.

    Returns:
        (mAP@0.5, mean precision, mean recall)
    """
    aps, precisions, recalls = [], [], []

    for image_id, gt in targets.items():
        boxes, scores = predictions.get(image_id, (np.zeros((0, 4)), np.zeros((0,))))

        if len(boxes) == 0:
            aps.append(0.0)
            precisions.append(0.0)
            recalls.append(0.0)
            continue

        order = np.argsort(-scores)
        boxes = boxes[order]
        scores = scores[order]

        if len(gt) == 0:
            aps.append(0.0)
            precisions.append(0.0)
            recalls.append(0.0)
            continue

        ious = iou_matrix(boxes, gt)
        matched = np.zeros(len(gt), dtype=bool)

        tp = np.zeros(len(boxes), dtype=np.float32)
        fp = np.zeros(len(boxes), dtype=np.float32)

        for i in range(len(boxes)):
            best = int(np.argmax(ious[i])) if len(gt) else -1
            if best >= 0 and ious[i, best] >= iou_threshold and not matched[best]:
                tp[i] = 1.0
                matched[best] = True
            else:
                fp[i] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        n_gt = len(gt)
        recall = tp_cum / max(n_gt, 1)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-8)

        aps.append(average_precision(recall, precision))
        precisions.append(float(precision[-1]))
        recalls.append(float(recall[-1]))

    if not aps:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(aps)), float(np.mean(precisions)), float(np.mean(recalls))


@torch.no_grad()
def run(model: MedSRDetInference, loader: DataLoader, device: str,
        hr_size: int = 640, conf_thres: float = 0.001,
        iou_thres: float = 0.6, max_det: int = 300):
    """Collect predictions and ground truth for one split."""
    from ultralytics.utils.ops import non_max_suppression

    model.eval()
    predictions, targets = {}, {}

    img_id = 0
    for data in loader:
        lr = data["lr"].to(device)
        det = model(lr)

        # det: {'cls': [B, nc, H, W] per level, 'bbox': [B, 4, H, W] per level}
        # Decode the finest level for evaluation; anchors are the pixel grid.
        cls, box = det["cls"][0], det["bbox"][0]
        b, _, h, w = cls.shape
        stride = hr_size / h

        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=device), torch.arange(w, device=device),
            indexing="ij",
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).float()          # [H,W,2]

        preds = torch.cat([box.permute(0, 2, 3, 1),                    # [B,H,W,4]
                           cls.permute(0, 2, 3, 1).sigmoid().max(-1, keepdim=True)[0]],
                          dim=-1)                                       # [B,H,W,5]

        flat = preds.reshape(b, h * w, 5)
        flat[..., 0:2] = (flat[..., 0:2] + grid.reshape(1, h * w, 2)) * stride
        flat[..., 2:4] = flat[..., 2:4] * stride

        nms_in = torch.cat([flat[..., :4], flat[..., 4:5]], dim=-1)
        nms_out = non_max_suppression(nms_in, conf_thres, iou_thres,
                                      max_det=max_det)

        for bi, det_i in enumerate(nms_out):
            key = img_id
            img_id += 1

            if det_i is not None and len(det_i):
                xyxy = det_i[:, :4].cpu().numpy().astype(np.float32)
                scores = det_i[:, 4].cpu().numpy().astype(np.float32)
            else:
                xyxy = np.zeros((0, 4), dtype=np.float32)
                scores = np.zeros((0,), dtype=np.float32)

            predictions[key] = (xyxy, scores)

            gt = data["boxes"][data["batch_idx"] == bi].cpu().numpy()
            if len(gt):
                # YOLO cx cy w h (normalised) -> xyxy pixels
                cx, cy, bw, bh = gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3]
                gt_xyxy = np.stack([
                    (cx - bw / 2) * hr_size, (cy - bh / 2) * hr_size,
                    (cx + bw / 2) * hr_size, (cy + bh / 2) * hr_size,
                ], axis=1).astype(np.float32)
            else:
                gt_xyxy = np.zeros((0, 4), dtype=np.float32)
            targets[key] = gt_xyxy

    return predictions, targets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dataset", default="brats2021",
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

    preds, targets = run(model, loader, args.device, hr_size=hr_size)
    map50, precision, recall = compute_map50(preds, targets)

    print(json_report := {
        "dataset": args.dataset,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "mAP@0.5": round(map50, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
