"""Shared detection decoding, post-processing and mAP@0.5.

A single implementation of the MedHead box parameterisation is used by
training (CIoU targets), evaluation and inference, so the box interpretation
can never diverge between the loss and the reported metrics.

Box parameterisation (manuscript Appendix E)
--------------------------------------------
MedHead regresses four channels ``(dx, dy, dw, dh)`` per grid cell.  The
decoding rule below is the standard anchor-free formulation; it is an
IMPLEMENTATION DETAIL not pinned down by the manuscript and is stated here for
reproducibility::

    bx = (sigmoid(dx) * 2 - 0.5 + gx) * stride      # centre x
    by = (sigmoid(dy) * 2 - 0.5 + gy) * stride      # centre y
    bw = (sigmoid(dw) * 2) ** 2 * stride            # width
    bh = (sigmoid(dh) * 2) ** 2 * stride            # height

Evaluation
----------
``compute_map50`` implements the standard dataset-level, class-aware object
detection AP@0.5: predictions from the whole dataset are collected per class,
ranked by confidence, matched to same-image / same-class ground truth at
IoU >= 0.5 (each GT matched at most once), and the AP of every class that has
ground-truth instances is averaged (COCO 101-point interpolation).
"""

from __future__ import annotations

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Box decoding (single source of truth for train / eval / inference)
# --------------------------------------------------------------------------- #
def decode_level_boxes(box_lvl: torch.Tensor, stride: float) -> torch.Tensor:
    """Decode one pyramid level from (dx, dy, dw, dh) to xyxy pixel boxes.

    Args:
        box_lvl: [B, 4, H, W] raw regression output.
        stride: pyramid stride of this level (8 / 16 / 32).

    Returns:
        [B, H, W, 4] xyxy boxes in input-resolution pixels.
    """
    B, _, H, W = box_lvl.shape
    gy, gx = torch.meshgrid(
        torch.arange(H, device=box_lvl.device),
        torch.arange(W, device=box_lvl.device),
        indexing="ij",
    )
    grid = torch.stack([gx, gy], dim=-1).float()                  # [H, W, 2]

    p = box_lvl.permute(0, 2, 3, 1)                               # [B, H, W, 4]
    bx = (torch.sigmoid(p[..., 0]) * 2.0 - 0.5 + grid[..., 0]) * stride
    by = (torch.sigmoid(p[..., 1]) * 2.0 - 0.5 + grid[..., 1]) * stride
    bw = (torch.sigmoid(p[..., 2]) * 2.0) ** 2 * stride
    bh = (torch.sigmoid(p[..., 3]) * 2.0) ** 2 * stride
    return torch.stack([bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2],
                       dim=-1)                                    # [B, H, W, 4]


def decode_medhead_outputs(det_out: dict, hr_size: int = 640,
                           strides=(8, 16, 32)):
    """Decode all pyramid levels of a MedHead output dict.

    Args:
        det_out: {'cls': [lvl0, lvl1, lvl2] of [B, nc, H, W],
                  'bbox': [lvl0, lvl1, lvl2] of [B, 4, H, W]}
        hr_size: input resolution (for stride inference sanity check).
        strides: strides aligned with the level list.

    Returns:
        List (one entry per level) of ``(boxes_xyxy [B, H*W, 4],
        class_scores [B, H*W, nc])``.
    """
    decoded = []
    for lvl, (cls_lvl, box_lvl) in enumerate(zip(det_out["cls"], det_out["bbox"])):
        H = cls_lvl.shape[-2]
        stride = hr_size / H
        # Optional consistency check against the declared strides.
        if lvl < len(strides) and abs(stride - strides[lvl]) > 0.51:
            raise ValueError(
                f"level {lvl}: inferred stride {stride} mismatches declared "
                f"stride {strides[lvl]}"
            )
        boxes = decode_level_boxes(box_lvl, stride).reshape(
            box_lvl.shape[0], -1, 4)
        scores = cls_lvl.sigmoid() \
            .permute(0, 2, 3, 1).reshape(cls_lvl.shape[0], -1, cls_lvl.shape[1])
        decoded.append((boxes, scores))
    return decoded


# --------------------------------------------------------------------------- #
# Multi-scale merge + class-aware NMS
# --------------------------------------------------------------------------- #
def postprocess_detections(det_out: dict, hr_size: int = 640,
                           conf_thres: float = 0.001, iou_thres: float = 0.6,
                           max_det: int = 300, strides=(8, 16, 32)) -> list:
    """Merge P3/P4/P5, threshold, and apply class-aware NMS.

    Returns:
        List of B dicts ``{"boxes": [N,4] xyxy float32, "scores": [N],
        "labels": [N] int64}``.
    """
    import torchvision

    decoded = decode_medhead_outputs(det_out, hr_size, strides)
    B = decoded[0][0].shape[0]
    results = []

    for b in range(B):
        boxes_all, scores_all, labels_all = [], [], []
        for boxes, scores in decoded:                 # boxes [B,HW,4] scores [B,HW,nc]
            b_boxes, b_scores = boxes[b], scores[b]
            conf, lab = b_scores.max(dim=1)           # best class per anchor
            keep = conf > conf_thres
            if keep.any():
                boxes_all.append(b_boxes[keep])
                scores_all.append(conf[keep])
                labels_all.append(lab[keep])

        if not boxes_all:
            results.append({"boxes": np.zeros((0, 4), np.float32),
                            "scores": np.zeros((0,), np.float32),
                            "labels": np.zeros((0,), np.int64)})
            continue

        boxes_cat = torch.cat(boxes_all)
        scores_cat = torch.cat(scores_all)
        labels_cat = torch.cat(labels_all)

        keep = torchvision.ops.batched_nms(boxes_cat, scores_cat, labels_cat,
                                           iou_thres)[:max_det]
        results.append({
            "boxes": boxes_cat[keep].detach().cpu().numpy().astype(np.float32),
            "scores": scores_cat[keep].detach().cpu().numpy().astype(np.float32),
            "labels": labels_cat[keep].detach().cpu().numpy().astype(np.int64),
        })
    return results


# --------------------------------------------------------------------------- #
# Dataset-level class-aware mAP@0.5
# --------------------------------------------------------------------------- #
def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area_a = np.clip(a[:, 2:] - a[:, :2], 0, None).prod(axis=1)
    area_b = np.clip(b[:, 2:] - b[:, :2], 0, None).prod(axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-8)


def _average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO-style AP with 101-point interpolation."""
    if len(recall) == 0:
        return 0.0
    order = np.argsort(recall)
    r, p = recall[order], precision[order]
    mrec = np.concatenate(([0.0], r, [1.0]))
    mpre = np.concatenate(([0.0], p, [0.0]))
    for i in range(mpre.size - 1, 0, -1):          # monotone decreasing envelope
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_map50(predictions: dict, targets: dict,
                  iou_threshold: float = 0.5):
    """Dataset-level, class-aware mAP@0.5.

    Args:
        predictions: {image_id: {"boxes": [N,4] xyxy px, "scores": [N],
                                 "labels": [N]}}
        targets:     {image_id: {"boxes": [M,4] xyxy px, "labels": [M]}}

    Returns:
        (mAP@0.5, micro precision, micro recall).  mAP is the mean AP over
        every class that has at least one ground-truth instance in the split;
        VinDr-CXR classes are therefore never merged into a single lesion
        class.
    """
    classes = set()
    for t in targets.values():
        classes.update(int(c) for c in t["labels"])

    aps, tp_total, fp_total, n_gt_total = [], 0, 0, 0

    for c in sorted(classes):
        # Collect (score, is_tp, img) over the whole dataset for this class.
        entries = []                                # (img_id, score, iou_best, gt_idx)
        n_gt = 0
        for img_id, t in targets.items():
            gt_lab = np.asarray(t["labels"])
            sel = np.where(gt_lab == c)[0]
            gt_boxes = np.asarray(t["boxes"], np.float32)[sel]
            n_gt += len(sel)

            pred = predictions.get(
                img_id, {"boxes": np.zeros((0, 4), np.float32),
                         "scores": np.zeros((0,), np.float32),
                         "labels": np.zeros((0,), np.int64)})
            plab = np.asarray(pred["labels"])
            psel = np.where(plab == c)[0]
            p_boxes = np.asarray(pred["boxes"], np.float32)[psel]
            p_scores = np.asarray(pred["scores"], np.float32)[psel]

            ious = _iou_matrix(p_boxes, gt_boxes)
            for i in range(len(p_scores)):
                best = int(np.argmax(ious[i])) if len(sel) else -1
                entries.append((img_id, float(p_scores[i]),
                                float(ious[i, best]) if best >= 0 else 0.0,
                                best))

        if n_gt == 0:
            continue

        entries.sort(key=lambda e: -e[1])
        matched = {}                                # (img_id, gt_idx) -> True
        tp = np.zeros(len(entries), np.float32)
        fp = np.zeros(len(entries), np.float32)
        for i, (img_id, _, iou, g) in enumerate(entries):
            if g >= 0 and iou >= iou_threshold and (img_id, g) not in matched:
                tp[i] = 1.0
                matched[(img_id, g)] = True
            else:
                fp[i] = 1.0

        tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
        recall = tp_cum / n_gt
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-8)
        aps.append(_average_precision(recall, precision))

        tp_total += float(tp_cum[-1]) if len(tp_cum) else 0.0
        fp_total += float(fp_cum[-1]) if len(fp_cum) else 0.0
        n_gt_total += n_gt

    if not aps:
        return float("nan"), float("nan"), float("nan")

    mAP = float(np.mean(aps))
    micro_p = tp_total / max(tp_total + fp_total, 1e-8)
    micro_r = tp_total / max(n_gt_total, 1e-8)
    return mAP, float(micro_p), float(micro_r)
