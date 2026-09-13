"""MedSRDet joint training (Section 4.3, Appendix E).

The complete objective optimised within every iteration (Eq. 5/6)::

    L_total = gamma_sr * L_SR + gamma_cmcl * L_CMCL + gamma_det * L_Det

with gamma_sr = gamma_cmcl = gamma_det = 1.0 in the primary experiments and::

    L_Det = lambda_cls * L_cls + lambda_box * L_box,   lambda_cls = lambda_box = 1.0

The gamma coefficients are applied OUTSIDE the detection loss; they are never
folded into lambda_cls / lambda_box.

Design points that follow the manuscript
----------------------------------------
* Mixed-modal mini-batches of EXACTLY ``batch_size`` samples: every batch
  jointly samples BraTS2021 (MRI), LUNA16 (CT) and VinDr-CXR (X-ray); the
  extra sample when ``batch_size`` is not divisible by three rotates across
  datasets (5+5+6 / 6+5+5 / 5+6+5) (Section 3.3).
* CMCL participates in the joint objective: for EVERY annotated lesion
  instance the Appendix C pipeline (``scripts/lesion_views.py``) builds an
  augmented view and an MFASR-enhanced view from the same lesion-centred
  support; the InfoNCE loss is non-zero in the training logs and regularises
  the shared encoder.  Key branch = full momentum encoder pathway of Table 3:
  ``k = g_phi_m(E_theta_m(x_sr))``, no gradients, EMA update (m = 0.999) after
  every optimiser step, FIFO memory queue K = 65,536.
* Dataset-specific label spaces: mixed-modal batches keep the source dataset
  of every sample after collation.  The shared encoder and the MedHead feature
  processing are shared; classification is routed to the per-dataset head
  (BraTS/LUNA: 1 channel, VinDr: 14 channels) and per-dataset losses are
  aggregated with a sample-weighted mean (Eq. 4, Appendix E).  No unified
  cross-modal disease taxonomy exists.
* Ground-truth class ids: VinDr-CXR keeps its native 14-class labels
  (``scripts/preprocess.py``); classification targets are one-hot in the
  dataset's own label space — 14 channels are never averaged into one.
* Fixed patient-level 7:1:2 split (Section 4.1): generated once by
  ``scripts/preprocess.py`` under an INDEPENDENT fixed split seed and
  persisted as ``splits/{train,val,test}_patients.txt``; the optimisation
  seeds 42-46 never change the data partition — every repeated run reads the
  identical persisted manifests (verified by ``verify_fixed_splits`` and the
  smoke tests).
* Negative sampling (Appendix D.4): 1:1 positive:negative re-randomised per
  epoch applies ONLY to BraTS2021 / LUNA16 *training* slices; validation and
  test keep every eligible slice; VinDr-CXR keeps its image-level annotations
  as-is.
* CMCL is training-time only: the projection head, momentum encoder and memory
  queue are absent from the inference graph.
* Optimisation: AdamW, lr 1e-4, weight decay 1e-2, cosine annealing,
  300 epochs, batch size 16, AMP enabled (Table 5).
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
from modules.detection_utils import compute_map50, decode_level_boxes
from modules.yolo_encoder import YOLOv12FeatureEncoder
from scripts.lesion_views import build_cmcl_views


# --------------------------------------------------------------------------- #
# Reproducibility (Section 4.1: five seeds, fixed split)
# --------------------------------------------------------------------------- #
VALID_SEEDS = {42, 43, 44, 45, 46}
NEGATIVE_SAMPLING_DATASETS = {"brats2021", "luna16"}


def seed_everything(seed: int) -> None:
    """Seed the OPTIMISATION only.

    Section 4.1: the five random seeds 42-46 vary stochastic optimisation;
    the data partition is a FIXED patient-level 7:1:2 split persisted by
    ``scripts/preprocess.py`` and is NEVER regenerated here — every repeated
    run (any seed in VALID_SEEDS) reads the identical split manifests.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def verify_fixed_splits(data_root: Path, dataset_names: list) -> None:
    """Assert the persisted split manifests are complete and disjoint.

    Training never regenerates splits: ``SliceDataset`` reads only the
    manifests written by ``scripts/preprocess.py``.  This guard fails loudly
    if a manifest is missing, empty, or leaks patients across subsets.
    """
    for d in dataset_names:
        split_dir = Path(data_root) / d / "splits"
        parts = {}
        for s in ("train", "val", "test"):
            f = split_dir / f"{s}_patients.txt"
            if not f.exists():
                raise FileNotFoundError(
                    f"{f} not found — the fixed patient-level split manifest "
                    "must be persisted by scripts/preprocess.py before "
                    "training (training seeds 42-46 never change the split)."
                )
            parts[s] = {line.strip() for line in f.read_text().splitlines()
                        if line.strip()}
            if not parts[s]:
                raise ValueError(f"{f} is empty")
        if not (parts["train"] & parts["val"]).issubset({}) or \
           (parts["train"] & parts["test"]) or (parts["val"] & parts["test"]):
            raise ValueError(
                f"{d}: patient leakage across train/val/test subsets"
            )
        print(f"[splits] {d}: train={len(parts['train'])} "
              f"val={len(parts['val'])} test={len(parts['test'])} "
              f"(disjoint, persisted manifest)")


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class SliceDataset(Dataset):
    """One prepared dataset (HR 640 / LR 160 / YOLO labels).

    Manuscript negative-sampling protocol (Appendix D.4) is encoded by
    ``balance_negatives``:

    * BraTS2021 / LUNA16 + ``split == "train"``  -> 1:1 positive:negative
      re-randomised every epoch;
    * validation / test of those datasets        -> keep every eligible slice;
    * VinDr-CXR                                  -> keep all image-level
      annotations (no 1:1 slice balancing).

    ``balance_negatives=None`` (default) selects the protocol automatically
    from ``dataset_name`` and ``split``; an explicit value overrides it.
    """

    def __init__(self, root: Path, split: str, dataset_name: str = "",
                 hr_size: int = 640, lr_size: int = 160, seed: int = 42,
                 balance_negatives: bool | None = None):
        self.root = Path(root)
        self.split = split
        self.dataset_name = dataset_name or self.root.name
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.seed = seed
        self.epoch = 0
        if balance_negatives is None:
            balance_negatives = (self.dataset_name in NEGATIVE_SAMPLING_DATASETS
                                 and split == "train")
        self.balance_negatives = bool(balance_negatives)

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
        """Re-randomise the negative slice pool (Appendix D.4, train only)."""
        self.epoch = epoch
        rng = random.Random(self.seed + epoch)
        rng.shuffle(self.negatives)

    def __len__(self) -> int:
        if self.balance_negatives:
            # 1:1 positive:negative balance, training side only.
            return len(self.positives) + min(len(self.negatives), len(self.positives))
        # Validation / test (and VinDr-CXR): keep every eligible slice.
        return len(self.positives) + len(self.negatives)

    def __getitem__(self, idx: int):
        import cv2

        if self.balance_negatives:
            if idx < len(self.positives):
                lr_p, hr_p, lbl_p = self.positives[idx]
            else:
                j = idx - len(self.positives)
                lr_p, hr_p, lbl_p = self.negatives[j % len(self.negatives)]
        else:
            if idx < len(self.positives):
                lr_p, hr_p, lbl_p = self.positives[idx]
            else:
                lr_p, hr_p, lbl_p = self.negatives[idx - len(self.positives)]

        lr = cv2.imread(str(lr_p), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        hr = cv2.imread(str(hr_p), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0

        if lr.ndim == 2:
            lr = np.repeat(lr[:, :, None], 3, axis=2)
        if hr.ndim == 2:
            hr = np.repeat(hr[:, :, None], 3, axis=2)

        boxes, class_ids = [], []
        text = Path(lbl_p).read_text().strip()
        for line in text.splitlines():
            p = line.split()
            if len(p) == 5:
                class_ids.append(int(p[0]))          # native dataset class id
                boxes.append([float(v) for v in p[1:]])

        return {
            "lr": torch.from_numpy(np.ascontiguousarray(lr)).permute(2, 0, 1).float(),
            "hr": torch.from_numpy(np.ascontiguousarray(hr)).permute(2, 0, 1).float(),
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "classes": torch.as_tensor(class_ids, dtype=torch.long),
            "dataset": self.dataset_name,
        }


def collate(samples):
    """Collate that preserves dataset identity and class ids."""
    lr = torch.stack([s["lr"] for s in samples])
    hr = torch.stack([s["hr"] for s in samples])
    boxes, classes, batch_idx = [], [], []
    for i, s in enumerate(samples):
        for bi, b in enumerate(s["boxes"]):
            boxes.append(b)
            classes.append(int(s["classes"][bi]))
            batch_idx.append(i)
    if boxes:
        boxes = torch.stack(boxes)
        classes = torch.as_tensor(classes, dtype=torch.long)
        batch_idx = torch.as_tensor(batch_idx, dtype=torch.long)
    else:
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        classes = torch.zeros((0,), dtype=torch.long)
        batch_idx = torch.zeros((0,), dtype=torch.long)
    return {
        "lr": lr, "hr": hr, "boxes": boxes, "classes": classes,
        "batch_idx": batch_idx,
        "source_dataset": [s["dataset"] for s in samples],
    }


class MixedModalBatchSampler(torch.utils.data.Sampler):
    """Mini-batches of EXACTLY ``batch_size`` samples covering all datasets.

    With three datasets and batch 16 the quotas are 5+5+6; the extra sample
    rotates across datasets batch by batch (5+5+6, 6+5+5, 5+6+5, ...) so no
    dataset is systematically favoured.  ``len(batch) == batch_size`` holds
    for every batch.
    """

    def __init__(self, datasets: list, batch_size: int, seed: int = 42):
        self.datasets = datasets
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        if batch_size < len(datasets):
            raise ValueError("batch_size must be >= number of datasets")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _quotas(self, batch_no: int) -> list:
        k = len(self.datasets)
        base = self.batch_size // k
        extra = self.batch_size - base * k
        quotas = [base] * k
        for e in range(extra):
            quotas[(batch_no + e) % k] += 1
        return quotas

    def __iter__(self):
        rng = random.Random(self.seed * 1000 + self.epoch)
        base = self.batch_size // len(self.datasets)
        # Every dataset can receive the rotating extra sample, so each needs
        # at least (base + 1) items for a batch to always be full-size.
        n_batches = min(len(d) for d in self.datasets) // (base + 1)
        n_batches = max(n_batches, 1)

        batches = []
        for b in range(n_batches):
            quotas = self._quotas(b)
            batch = []
            for di, quota in enumerate(quotas):
                idx = [rng.randrange(len(self.datasets[di])) for _ in range(quota)]
                batch.extend([(di, i) for i in idx])
            rng.shuffle(batch)
            assert len(batch) == self.batch_size, (
                f"mixed-modal batch must contain exactly {self.batch_size} "
                f"samples, got {len(batch)}"
            )
            batches.append(batch)
        return iter(batches)

    def __len__(self):
        base = self.batch_size // len(self.datasets)
        return max(1, min(len(d) for d in self.datasets) // (base + 1))


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class MedSRDet(nn.Module):
    """MFASR + shared YOLOv12n encoder + MedHead + CMCL (training-time).

    ``encoder`` allows injecting a pre-built feature encoder (used by the
    smoke tests); by default the shared YOLOv12n encoder is constructed from
    the ultralytics checkpoint given in the config.
    """

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

    def encode(self, x):
        """Return the P3/P4/P5 feature maps of the shared encoder."""
        return self.shared_encoder(x)

    def forward(self, lr, dataset: str = "brats2021"):
        """Inference pathway: MFASR -> encoder -> MedHead (CMCL excluded)."""
        sr = self.mfasr(lr)
        p345 = self.encode(sr)
        return self.medhead(p345, dataset=dataset)

    # ------------------------------------------------------------------ #
    def forward_losses(self, lr, hr, boxes, classes, batch_idx,
                       source_datasets, sr_criterion,
                       gamma_sr, gamma_cmcl, gamma_det, cmcl_views=None):
        """Joint objective of Eq. (5)/(6) for one mixed-modal mini-batch."""
        sr = self.mfasr(lr)
        loss_sr, sr_items = sr_criterion(sr, hr)

        p345 = self.encode(sr)
        feats = self.medhead.feature_refine(p345)     # shared feature processing

        # ---- dataset-specific detection losses, sample-weighted mean (Eq. 4)
        B = lr.shape[0]
        unique_ds = list(dict.fromkeys(source_datasets))
        loss_det = lr.new_zeros(())
        for d in unique_ds:
            sel_samples = [i for i, name in enumerate(source_datasets)
                           if name == d]
            idx = torch.as_tensor(sel_samples, device=lr.device)
            feats_d = [f.index_select(0, idx) for f in feats]

            mask = torch.zeros(boxes.shape[0], dtype=torch.bool,
                               device=boxes.device)
            remap = {}
            for local_b, glob_b in enumerate(sel_samples):
                mask |= batch_idx == glob_b
                remap[glob_b] = local_b
            boxes_d = boxes[mask]
            classes_d = classes[mask]
            batch_d = torch.as_tensor(
                [remap[int(v)] for v in batch_idx[mask].tolist()],
                device=boxes.device, dtype=torch.long)

            cls_d = self.medhead.classify(feats_d, d)     # per-dataset head
            box_d = self.medhead.regress(feats_d)         # shared regression
            l_d = detection_loss(cls_d, box_d, boxes_d, classes_d, batch_d,
                                 hr_size=sr.shape[-1])    # SR/HR pixel space
            loss_det = loss_det + (len(sel_samples) / B) * l_d

        # ---- CMCL on the Appendix-C lesion views (Table 3 momentum pathway)
        if cmcl_views is not None:
            with torch.no_grad():
                # Key view: MFASR-enhanced; produced by the current MFASR and
                # encoded by the frozen momentum encoder (no gradients on the
                # key branch, Table 3).
                key_imgs = self.mfasr(cmcl_views["low_res"])
            loss_cmcl = self.cmcl.forward_from_images(
                cmcl_views["augmented"], key_imgs, self.shared_encoder)
        else:
            loss_cmcl = torch.zeros((), device=sr.device)

        loss_total = (gamma_sr * loss_sr + gamma_cmcl * loss_cmcl
                      + gamma_det * loss_det)

        return {
            "loss_total": loss_total,
            "loss_sr": loss_sr,
            "loss_cmcl": loss_cmcl,
            "loss_det": loss_det,
            "sr_items": sr_items,
            "sr": sr,
        }


def detection_loss(cls_levels, box_levels, boxes, classes, batch_idx,
                   hr_size: int = 640, lambda_cls: float = 1.0,
                   lambda_box: float = 1.0):
    """L_Det = lambda_cls * L_cls + lambda_box * L_box (Appendix E).

    Classification: BCE over every grid cell and every channel of the
    dataset's OWN label space.  The target tensor has shape [B, C, H, W]; each
    ground-truth object sets the positive target in the channel of its native
    ``class_id`` — 14-class label spaces are never averaged into one channel.

    Regression: CIoU on the assigned cells only, decoded with the SAME anchor-
    free parameterisation as evaluation/inference
    (``modules.detection_utils.decode_level_boxes``).

    Per-dataset losses are averaged over the samples of the dataset subset;
    the caller aggregates the subsets with a sample-weighted mean (Eq. 4).
    """
    from ultralytics.utils.metrics import bbox_iou

    device = cls_levels[0].device
    cls_terms, box_terms = [], []

    gt_xyxy = boxes.new_zeros((boxes.shape[0], 4))
    if boxes.numel():
        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        gt_xyxy = torch.stack([
            (cx - bw / 2) * hr_size, (cy - bh / 2) * hr_size,
            (cx + bw / 2) * hr_size, (cy + bh / 2) * hr_size,
        ], dim=1)

    for cls_lvl, box_lvl in zip(cls_levels, box_levels):
        B, C, H, W = cls_lvl.shape
        stride = hr_size / H

        cls_target = torch.zeros((B, C, H, W), device=device)
        box_target = torch.zeros((B, H, W, 4), device=device)
        assigned = torch.zeros((B, H, W), device=device, dtype=torch.bool)

        for b in range(B):
            sel = (batch_idx == b).nonzero(as_tuple=True)[0]
            if len(sel) == 0:
                continue
            gt = gt_xyxy[sel]
            cx_px = (gt[:, 0] + gt[:, 2]) / 2.0
            cy_px = (gt[:, 1] + gt[:, 3]) / 2.0
            gi = (cx_px / stride).long().clamp(0, W - 1)
            gj = (cy_px / stride).long().clamp(0, H - 1)
            cid = classes[sel].clamp(0, C - 1).long()
            cls_target[b, cid, gj, gi] = 1.0
            box_target[b, gj, gi] = gt
            assigned[b, gj, gi] = True

        # ---- classification: BCE in the dataset-specific label space ----
        cls_terms.append(
            torch.nn.functional.binary_cross_entropy_with_logits(
                cls_lvl.float(), cls_target.float()
            )
        )

        # ---- regression: CIoU on the assigned cells only ----
        pred_xyxy = decode_level_boxes(box_lvl, stride)      # [B,H,W,4]
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
    dataset_names = ["brats2021", "luna16", "vindrcxr"]
    dataset_nc = {d: cfg["datasets"][d]["num_classes"] for d in dataset_names}

    # Section 4.1: fixed patient-level 7:1:2 split — verified once, then read
    # identically by every repeated run (training seeds 42-46).
    verify_fixed_splits(root, dataset_names)

    common = dict(hr_size=cfg["common"]["hr_size"],
                  lr_size=cfg["common"]["lr_size"], seed=args.seed)
    train_sets = [SliceDataset(root / d, "train", dataset_name=d, **common)
                  for d in dataset_names]
    # Validation keeps every eligible slice and its dataset identity.
    val_sets = [(d, SliceDataset(root / d, "val", dataset_name=d, **common))
                for d in dataset_names]

    sampler = MixedModalBatchSampler(train_sets, batch_size, seed=args.seed)

    model = MedSRDet(cfg, dataset_nc).to(args.device)

    # Table 3 momentum encoder E_theta_m: attach BEFORE the optimiser is
    # built; the copy is frozen (requires_grad=False) and updated by EMA.
    model.cmcl.attach_momentum_backbone(model.shared_encoder)

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
    amp_on = bool(t["amp"]) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_on and amp_dtype is torch.float16))

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
            boxes = data["boxes"].to(args.device)
            classes = data["classes"].to(args.device)
            batch_idx = data["batch_idx"].to(args.device)

            # Appendix C: one positive pair per annotated lesion instance.
            cmcl_views = None
            if boxes.shape[0] > 0:
                aug_v, low_v = build_cmcl_views(
                    hr, data["boxes"], data["classes"], data["batch_idx"],
                    seed=args.seed, epoch=epoch,
                )
                cmcl_views = {"augmented": aug_v, "low_res": low_v}

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype,
                                enabled=amp_on):
                out_d = model.forward_losses(
                    lr, hr, boxes, classes, batch_idx,
                    data["source_dataset"], sr_criterion,
                    t["gamma_sr"], t["gamma_cmcl"], t["gamma_det"],
                    cmcl_views=cmcl_views,
                )
                loss = out_d["loss_total"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Table 3: EMA update of E_theta_m / g_phi_m AFTER the optimiser
            # step:  theta_m <- m * theta_m + (1 - m) * theta.
            model.cmcl.update_momentum_encoder(model.shared_encoder)

            tot += float(loss.detach())
            sr_t += float(out_d["loss_sr"].detach())
            cmcl_t += float(out_d["loss_cmcl"].detach())
            det_t += float(out_d["loss_det"].detach())
            n_batches += 1

        scheduler.step()

        map50 = float("nan")
        if epoch % t.get("val_every", 1) == 0:
            map50 = evaluate_map(model, val_sets, args.device, batch_size,
                                 hr_size=cfg["common"]["hr_size"])

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

    ``val_sets`` is a list of ``(dataset_name, dataset)`` pairs so every
    validation pass is routed to the correct dataset-specific classification
    head.  The held-out test set is evaluated once, after checkpoint selection
    (Section 4.3).  CMCL is inactive in eval mode.
    """
    from scripts.evaluate import run

    model.eval()
    aps = []
    with torch.no_grad():
        for dataset_name, ds in val_sets:
            loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate,
                                shuffle=False)
            predictions, targets = run(model, loader, device,
                                       dataset=dataset_name, hr_size=hr_size)
            map50, _, _ = compute_map50(predictions, targets)
            if not math.isnan(map50):
                aps.append(map50)
    model.train()
    return float(np.mean(aps)) if aps else float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
