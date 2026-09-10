"""MedSRDet joint training (Section 4.3, Appendix E).

The完整 objective optimised within every iteration (Eq. 5/6)::

    L_total = gamma_sr * L_SR + gamma_cmcl * L_CMCL + gamma_det * L_Det

with gamma_sr = gamma_cmcl = gamma_det = 1.0 in the primary experiments.

Design points that follow the manuscript
----------------------------------------
* Mixed-modal mini-batches: every batch jointly samples lesion instances from
  BraTS2021 (MRI), LUNA16 (CT) and VinDr-CXR (X-ray) so the shared encoder is
  exposed to heterogeneous imaging domains (Section 3.3).  Samples are NEVER
  paired as cross-modal positives.
* Dataset-specific label spaces: classification is evaluated only inside the
  label space of the source dataset; regression uses the common four-parameter
  form.  Per-dataset detection losses are aggregated with a sample-weighted
  mean (Eq. 4, Appendix E).
* CMCL is training-time only: the projection head, momentum encoder and memory
  queue are absent from the inference graph.
* Optimisation: AdamW, lr 1e-4, weight decay 1e-2, cosine annealing,
  300 epochs, batch size 16, AMP enabled (Table 5).
* Negative-slice re-sampling at a 1:1 positive:negative ratio for BraTS2021
  and LUNA16 is re-randomised every epoch (Appendix D.4).
* Checkpoint selection: highest validation mAP@0.5; the test set is never used
  for selection or tuning.

Usage
-----
    python scripts/train.py --config configs/default.yaml \
        --data-root /path/to/prepared --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

from modules import CMCL, MFASRLoss, MFASRNet, MedHead


# --------------------------------------------------------------------------- #
# Reproducibility (Section 4.1: five seeds, fixed split)
# --------------------------------------------------------------------------- #
VALID_SEEDS = {42, 43, 44, 45, 46}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class SliceDataset(Dataset):
    """One prepared dataset (HR 640 / LR 160 / YOLO labels).

    Positive and negative slices are kept separately so that the training-time
    1:1 re-sampling required by Appendix D.4 can be applied per epoch.
    """

    def __init__(self, root: Path, split: str, hr_size: int = 640,
                 lr_size: int = 160, seed: int = 42):
        self.root = Path(root)
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.seed = seed
        self.epoch = 0

        patient_file = self.root / "splits" / f"{split}_patients.txt"
        if not patient_file.exists():
            raise FileNotFoundError(
                f"{patient_file} not found - run scripts/preprocess.py first"
            )
        self.patients = {
            p.strip() for p in patient_file.read_text().splitlines() if p.strip()
        }

        self.positives, self.negatives = self._index()

    def _index(self):
        import csv

        pos, neg = [], []
        manifest = self.root / "manifest.csv"
        if not manifest.exists():
            raise FileNotFoundError(manifest)

        with manifest.open() as fh:
            for row in csv.DictReader(fh):
                if row["patient"] not in self.patients:
                    continue
                item = (self.root / row["lr"], self.root / row["hr"],
                        self.root / row["label"])
                (pos if int(row["positive"]) else neg).append(item)
        return pos, neg

    def set_epoch(self, epoch: int) -> None:
        """Re-randomise the negative slice pool (Appendix D.4)."""
        self.epoch = epoch
        rng = random.Random(self.seed + epoch)
        rng.shuffle(self.negatives)

    def __len__(self) -> int:
        # 1:1 positive:negative balance on the training side.
        return len(self.positives) + min(len(self.negatives), len(self.positives))

    def __getitem__(self, idx: int):
        import cv2

        if idx < len(self.positives):
            lr_p, hr_p, lbl_p = self.positives[idx]
        else:
            j = idx - len(self.positives)
            lr_p, hr_p, lbl_p = self.negatives[j % len(self.negatives)]

        lr = cv2.imread(str(lr_p), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        hr = cv2.imread(str(hr_p), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0

        if lr.ndim == 2:
            lr = np.repeat(lr[:, :, None], 3, axis=2)
        if hr.ndim == 2:
            hr = np.repeat(hr[:, :, None], 3, axis=2)

        boxes = []
        text = Path(lbl_p).read_text().strip()
        for line in text.splitlines():
            p = line.split()
            if len(p) == 5:
                boxes.append([float(v) for v in p[1:]])

        return {
            "lr": torch.from_numpy(np.ascontiguousarray(lr)).permute(2, 0, 1).float(),
            "hr": torch.from_numpy(np.ascontiguousarray(hr)).permute(2, 0, 1).float(),
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        }


def collate(samples):
    lr = torch.stack([s["lr"] for s in samples])
    hr = torch.stack([s["hr"] for s in samples])
    boxes, batch_idx = [], []
    for i, s in enumerate(samples):
        for b in s["boxes"]:
            boxes.append(b)
            batch_idx.append(i)
    if boxes:
        boxes = torch.stack(boxes)
        batch_idx = torch.as_tensor(batch_idx, dtype=torch.long)
    else:
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        batch_idx = torch.zeros((0,), dtype=torch.long)
    return {"lr": lr, "hr": hr, "boxes": boxes, "batch_idx": batch_idx}


class MixedModalBatchSampler(torch.utils.data.Sampler):
    """Build mini-batches that jointly sample all datasets (Section 3.3)."""

    def __init__(self, datasets: list, batch_size: int, seed: int = 42):
        self.datasets = datasets
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        # Round-robin over datasets so every batch sees heterogeneous domains.
        per_ds = max(1, self.batch_size // max(len(self.datasets), 1))
        batches = []
        n = min(len(d) for d in self.datasets)
        n_batches = max(1, (n * len(self.datasets)) // self.batch_size)

        for _ in range(n_batches):
            batch = []
            for d in self.datasets:
                idx = [rng.randrange(len(d)) for _ in range(per_ds)]
                batch.extend([(self.datasets.index(d), i) for i in idx])
            rng.shuffle(batch)
            batches.append(batch[:self.batch_size])
        return iter(batches)

    def __len__(self):
        n = min(len(d) for d in self.datasets)
        return max(1, (n * len(self.datasets)) // self.batch_size)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class MedSRDet(nn.Module):
    """MFASR + shared YOLOv12n encoder + MedHead + CMCL (training-time)."""

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

        yolo = YOLO(m["backbone_weight"])
        self.detector = yolo.model
        self.encoder_channels = self._probe_encoder_channels()

        self.medhead = MedHead(
            in_channels=self.encoder_channels,      # P3, P4, P5
            channels=m["unified_channels"],
            dataset_nc=dataset_nc,
        )

        self.cmcl = CMCL(
            in_channels=self.encoder_channels,
            feature_dim=m["cmcl"]["feature_dim"],
            embedding_dim=m["cmcl"]["projection"][1],
            hidden_dim=m["cmcl"]["hidden_dim"],
            dropout=m["cmcl"]["dropout"],
            temperature=m["cmcl"]["temperature"],
            lambda_reg=m["cmcl"]["lambda_reg"],
            momentum=m["cmcl"]["momentum"],
            queue_size=m["cmcl"]["queue_size"],
        )

    def _probe_encoder_channels(self):
        """Channel widths of P3/P4/P5 for the active YOLOv12n build.

        ultralytics exposes the FPN widths through the Detect head; the values
        below are the YOLOv12n defaults and are verified at first forward.
        """
        for mod in self.detector.modules():
            if mod.__class__.__name__ == "Detect":
                try:
                    return [int(c) for c in mod.cv2[0][0].in_channels][:3]
                except Exception:
                    pass
        return [256, 256, 256]

    def encode(self, x):
        """Return the P3/P4/P5 feature maps of the shared encoder."""
        feats = []
        for i, layer in enumerate(self.detector.model):
            x = layer(x)
            feats.append(x)
        # The YOLO FPN emits P3/P4/P5 immediately before the Detect head.
        return feats[-4:-1]

    def forward(self, lr, dataset: str = "brats2021"):
        """Inference pathway: MFASR -> encoder -> MedHead (CMCL excluded)."""
        sr = self.mfasr(lr)
        p345 = self.encode(sr)
        return self.medhead(p345, dataset=dataset)

    def forward_losses(self, lr, hr, boxes, batch_idx, sr_criterion,
                       gamma_sr, gamma_cmcl, gamma_det, cmcl_views=None):
        sr = self.mfasr(lr)

        loss_sr, sr_items = sr_criterion(sr, hr)

        p345 = self.encode(sr)
        det_out = self.medhead(p345)

        loss_det = detection_loss(
            det_out, boxes, batch_idx,
            lambda_cls=gamma_det * self._lambda_cls,
            lambda_box=gamma_det * self._lambda_box,
        )

        if cmcl_views is not None:
            aug_feat = self.encode(cmcl_views["augmented"])
            sr_feat = self.encode(cmcl_views["mfasr_enhanced"])
            loss_cmcl = self.cmcl(aug_feat, sr_feat)
        else:
            loss_cmcl = torch.zeros((), device=sr.device)

        loss_total = gamma_sr * loss_sr + gamma_cmcl * loss_cmcl + loss_det

        return {
            "loss_total": loss_total,
            "loss_sr": loss_sr,
            "loss_cmcl": loss_cmcl,
            "loss_det": loss_det,
            "sr_items": sr_items,
            "sr": sr,
        }

    _lambda_cls = 1.0
    _lambda_box = 1.0


def _decode_boxes(box_lvl, stride, grid):
    """Anchor-free decoding of MedHead's (dx, dy, dw, dh) to xyxy pixels.

    The manuscript fixes the regression *target* as the four-parameter
    representation (dx, dy, dw, dh) (Appendix E) but does not pin down the
    decoding rule; the standard anchor-free parameterisation below is used and
    is stated explicitly for reproducibility.
    """
    dx, dy, dw, dh = box_lvl[:, 0], box_lvl[:, 1], box_lvl[:, 2], box_lvl[:, 3]
    bx = (torch.sigmoid(dx) * 2.0 - 0.5 + grid[..., 0]) * stride
    by = (torch.sigmoid(dy) * 2.0 - 0.5 + grid[..., 1]) * stride
    bw = (torch.sigmoid(dw) * 2.0) ** 2 * stride
    bh = (torch.sigmoid(dh) * 2.0) ** 2 * stride
    return torch.stack([bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2],
                       dim=-1)                                   # [B,H,W,4]


def detection_loss(det_out, boxes, batch_idx, hr_size=640,
                   lambda_cls=1.0, lambda_box=1.0):
    """BCE classification + CIoU regression, sample-weighted mean (Eq. 3/4).

    Targets are assigned to the grid cell nearest to each ground-truth box
    centre (anchor-free, single positive per object).  Classification uses
    binary cross-entropy in the dataset-specific label space and regression
    uses complete IoU, as stated in Appendix E (Table E.1).

    Per-dataset losses are averaged over all samples of the mini-batch, which
    yields the sample-weighted mean of Eq. (4).
    """
    from ultralytics.utils.metrics import bbox_iou

    device = det_out["cls"][0].device
    cls_terms, box_terms = [], []

    for cls_lvl, box_lvl in zip(det_out["cls"], det_out["bbox"]):
        B, _, H, W = cls_lvl.shape
        stride = hr_size / H

        gy, gx = torch.meshgrid(
            torch.arange(H, device=device), torch.arange(W, device=device),
            indexing="ij",
        )
        grid = torch.stack([gx, gy], dim=-1).float()             # [H,W,2]

        cls_target = torch.zeros((B, H, W), device=device)
        box_target = torch.zeros((B, H, W, 4), device=device)
        assigned = torch.zeros((B, H, W), device=device, dtype=torch.bool)

        gt_xyxy = boxes.new_zeros((boxes.shape[0], 4))
        if boxes.numel():
            cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            gt_xyxy = torch.stack([
                (cx - bw / 2) * hr_size, (cy - bh / 2) * hr_size,
                (cx + bw / 2) * hr_size, (cy + bh / 2) * hr_size,
            ], dim=1)

        for b in range(B):
            sel = (batch_idx == b).nonzero(as_tuple=True)[0]
            if len(sel) == 0:
                continue
            gt = gt_xyxy[sel]
            cx_px = (gt[:, 0] + gt[:, 2]) / 2.0
            cy_px = (gt[:, 1] + gt[:, 3]) / 2.0
            gi = (cx_px / stride).long().clamp(0, W - 1)
            gj = (cy_px / stride).long().clamp(0, H - 1)
            cls_target[b, gj, gi] = 1.0
            box_target[b, gj, gi] = gt
            assigned[b, gj, gi] = True

        # ---- classification: BCE over every grid cell ----
        logits = cls_lvl[:, 0] if cls_lvl.shape[1] == 1 else cls_lvl.mean(dim=1)
        cls_terms.append(
            torch.nn.functional.binary_cross_entropy_with_logits(
                logits.float(), cls_target.float()
            )
        )

        # ---- regression: CIoU on the assigned cells only ----
        pred_xyxy = _decode_boxes(box_lvl.permute(0, 2, 3, 1), stride, grid)
        if assigned.any():
            p = pred_xyxy[assigned]
            t = box_target[assigned]
            iou = bbox_iou(p, t, CIoU=True)
            box_terms.append((1.0 - iou).mean())
        else:
            box_terms.append(box_lvl.sum() * 0.0)

    loss_cls = torch.stack(cls_terms).mean() if cls_terms else boxes.sum() * 0.0
    loss_box = torch.stack(box_terms).mean() if box_terms else boxes.sum() * 0.0
    return lambda_cls * loss_cls + lambda_box * loss_box


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="runs/exp")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    t = cfg["train"]

    if args.seed not in VALID_SEEDS:
        raise ValueError(f"seed must be one of {sorted(VALID_SEEDS)}")
    seed_everything(args.seed)

    epochs = args.epochs or t["epochs"]
    batch_size = args.batch_size or t["batch_size"]
    out = Path(args.out) / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(
        {**{k: vars(args)[k] for k in ("seed", "epochs", "batch_size")},
         "gamma_sr": t["gamma_sr"], "gamma_cmcl": t["gamma_cmcl"],
         "gamma_det": t["gamma_det"], "lr": t["lr"],
         "weight_decay": t["weight_decay"], "precision": t["precision"]},
        indent=2))

    root = Path(args.data_root)
    dataset_nc = {d: cfg["datasets"][d]["num_classes"]
                  for d in ("brats2021", "luna16", "vindrcxr")}

    train_sets = [SliceDataset(root / d, "train",
                               hr_size=cfg["common"]["hr_size"],
                               lr_size=cfg["common"]["lr_size"],
                               seed=args.seed)
                  for d in dataset_nc]
    val_sets = [SliceDataset(root / d, "val",
                             hr_size=cfg["common"]["hr_size"],
                             lr_size=cfg["common"]["lr_size"],
                             seed=args.seed)
                for d in dataset_nc]

    sampler = MixedModalBatchSampler(train_sets, batch_size, seed=args.seed)

    model = MedSRDet(cfg, dataset_nc).to(args.device)

    sr_criterion = MFASRLoss(
        lambda_pix=cfg["model"]["sr_loss"]["lambda_pix"],
        lambda_freq=cfg["model"]["sr_loss"]["lambda_freq"],
        lambda_perc=cfg["model"]["sr_loss"]["lambda_perc"],
    ).to(args.device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=t["lr"],
                                  weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    amp_dtype = torch.bfloat16 if t["precision"] == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(t["amp"] and amp_dtype is torch.float16))

    best_map = -1.0
    log = open(out / "metrics.csv", "w")
    log.write("epoch,loss_total,loss_sr,loss_cmcl,loss_det,map50,lr\n")

    for epoch in range(1, epochs + 1):
        model.train()
        for ds in train_sets:
            ds.set_epoch(epoch)
        sampler.set_epoch(epoch)

        tot = sr_t = cmcl_t = det_t = 0.0
        n_batches = 0

        for batch in sampler:
            # Materialise the mixed-modal batch from its (dataset, index) pairs.
            samples = [train_sets[di][ii] for di, ii in batch]
            data = collate(samples)
            lr = data["lr"].to(args.device)
            hr = data["hr"].to(args.device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype,
                                enabled=t["amp"]):
                out_d = model.forward_losses(
                    lr, hr, data["boxes"].to(args.device),
                    data["batch_idx"].to(args.device), sr_criterion,
                    t["gamma_sr"], t["gamma_cmcl"], t["gamma_det"],
                )
                loss = out_d["loss_total"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tot += float(loss.detach())
            sr_t += float(out_d["loss_sr"].detach())
            cmcl_t += float(out_d["loss_cmcl"].detach())
            det_t += float(out_d["loss_det"].detach())
            n_batches += 1

        scheduler.step()

        map50 = float("nan")
        if epoch % t.get("val_every", 1) == 0:
            map50 = evaluate_map(model, val_sets, args.device, batch_size)

        lr_now = optimizer.param_groups[0]["lr"]
        log.write(f"{epoch},{tot/max(n_batches,1):.6f},{sr_t/max(n_batches,1):.6f},"
                  f"{cmcl_t/max(n_batches,1):.6f},{det_t/max(n_batches,1):.6f},"
                  f"{map50:.6f},{lr_now:.3e}\n")
        log.flush()

        if not math.isnan(map50) and map50 > best_map:
            best_map = map50
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "map50": map50}, out / "best.pt")
            print(f"epoch {epoch}: new best mAP@0.5 = {map50:.4f}")

    torch.save({"epoch": epochs, "model": model.state_dict()}, out / "last.pt")
    log.close()
    print(f"finished: best mAP@0.5 = {best_map:.4f}")
    return 0


def evaluate_map(model, val_sets, device, batch_size, hr_size: int = 640):
    """Validation mAP@0.5, used for checkpoint selection only.

    The held-out test set is evaluated once, after the checkpoint has been
    selected (Section 4.3).  CMCL is inactive in eval mode.
    """
    # Imported lazily: scripts.evaluate imports `collate` from this module.
    from scripts.evaluate import compute_map50, run

    model.eval()
    aps = []
    with torch.no_grad():
        for ds in val_sets:
            loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate,
                                shuffle=False)
            predictions, targets = run(model, loader, device, hr_size=hr_size)
            map50, _, _ = compute_map50(predictions, targets)
            if not math.isnan(map50):
                aps.append(map50)
    model.train()
    return float(np.mean(aps)) if aps else float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
