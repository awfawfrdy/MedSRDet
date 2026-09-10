"""Dataset preprocessing and 2D detection-target construction.

Implements Section 4.2 and Appendix D (Tables D.1) of the manuscript:

  D.1  common protocol         : 1.0 mm isotropic resampling, letterbox to
                                 640x640, intensities in [0, 1], LR target
                                 obtained by bicubic downsampling to 160x160.
  D.2  BraTS2021 (MRI)         : T1ce / T2 / FLAIR stacked as three channels,
                                 per-sequence z-score inside the non-zero brain
                                 region, clipped to [-5, 5], mapped to [0, 1];
                                 enhancing-tumor (ET, label 4) mask converted
                                 to axial connected components -> tight boxes.
  D.3  LUNA16 (CT)             : HU clipped to [-1000, 400] and mapped to
                                 [0, 1]; nodule centre/diameter converted to
                                 slice-wise cross-section boxes with
                                 r_s = sqrt(r^2 - dz^2).
  D.4  VinDr-CXR (X-ray)       : 0.5-99.5 percentile clipping, mapped to
                                 [0, 1], replicated to three channels; native
                                 2D boxes transformed with identical
                                 letterbox parameters.

Patient-level train/val/test partitioning is performed BEFORE slice extraction
so that no patient contributes to more than one subset.

Usage
-----
    python scripts/preprocess.py --dataset brats2021 \
        --raw /path/to/BraTS2021 --out /path/to/prepared --split-seed 42

Output layout (per dataset)::

    hr/images/<patient>/<patient>_z<zzz>.png     640x640
    lr/images/<patient>/<patient>_z<zzz>.png     160x160
    labels/<patient>/<patient>_z<zzz>.txt        YOLO cx cy w h (normalised)
    manifest.csv                                 slice index + positive flag
    splits/{train,val,test}_patients.txt         patient-level manifest
    protocol.json                                machine-readable record
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Common helpers (Appendix D.1)
# --------------------------------------------------------------------------- #
def letterbox(image: np.ndarray, boxes: np.ndarray, size: int = 640):
    """Aspect-ratio-preserving resize with padding; boxes follow the transform.

    Args:
        image: [H, W] or [H, W, C] array in [0, 1].
        boxes: [N, 4] array of (x1, y1, x2, y2) in original pixel coordinates.

    Returns:
        (padded [size, size, C] image, transformed boxes, (scale, pad_x, pad_y))
    """
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2

    canvas = np.zeros((size, size, *resized.shape[2:]), dtype=resized.dtype)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    if len(boxes):
        scaled = boxes.astype(np.float32) * scale
        scaled[:, [0, 2]] += pad_x
        scaled[:, [1, 3]] += pad_y
    else:
        scaled = boxes.astype(np.float32)

    return canvas, scaled, (scale, pad_x, pad_y)


def to_yolo(boxes: np.ndarray, size: int = 640) -> list:
    """Convert (x1, y1, x2, y2) to normalised YOLO cx cy w h."""
    out = []
    for x1, y1, x2, y2 in boxes:
        cx = ((x1 + x2) / 2.0) / size
        cy = ((y1 + y2) / 2.0) / size
        bw = (x2 - x1) / size
        bh = (y2 - y1) / size
        # Clamp to the valid normalised range.
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        bw, bh = min(max(bw, 0.0), 1.0), min(max(bh, 0.0), 1.0)
        if bw <= 0 or bh <= 0:
            continue
        out.append([cx, cy, bw, bh])
    return out


def make_lr(hr: np.ndarray, lr_size: int = 160) -> np.ndarray:
    """Bicubic downsampling of the HR target (Appendix D.1)."""
    return cv2.resize(hr, (lr_size, lr_size), interpolation=cv2.INTER_CUBIC)


def zscore_nonzero(volume: np.ndarray, clip: float = 5.0) -> np.ndarray:
    """z-score inside the non-zero region, clip to [-clip, clip], map to [0, 1].

    Used for every MRI sequence (Appendix D.2).
    """
    mask = volume > 0
    out = np.zeros_like(volume, dtype=np.float32)
    if not mask.any():
        return out
    vals = volume[mask].astype(np.float32)
    mu, sigma = vals.mean(), vals.std()
    if sigma < 1e-8:
        sigma = 1.0
    out[mask] = np.clip((vals - mu) / sigma, -clip, clip)
    return (out + clip) / (2.0 * clip)


def connected_component_boxes(mask: np.ndarray, connectivity: int = 8):
    """Tight axis-aligned bounding boxes of non-empty connected components.

    Appendix D.2, Eq. (D.1).  No context margin is added.
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    if not mask_u8.any():
        return []

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=connectivity
    )

    boxes = []
    for i in range(1, n):                      # skip background (index 0)
        x, y, w, h = (int(v) for v in stats[i][:4])
        if w <= 0 or h <= 0:
            continue
        boxes.append([x, y, x + w, y + h])
    return boxes


def patient_split(patient_ids, ratios=(0.7, 0.1, 0.2), seed: int = 42):
    """Patient-level 7:1:2 partition (Section 4.1)."""
    ids = sorted(patient_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))

    return {
        "train": sorted(ids[:n_train]),
        "val": sorted(ids[n_train:n_train + n_val]),
        "test": sorted(ids[n_train + n_val:]),
    }


# --------------------------------------------------------------------------- #
# BraTS2021 (Appendix D.2)
# --------------------------------------------------------------------------- #
def prepare_brats2021(raw: Path, out: Path, split_seed: int,
                      sequences=("t1ce", "t2", "flair"),
                      et_label: int = 4, hr_size: int = 640,
                      lr_size: int = 160) -> dict:
    """Build the three-channel MRI input and ET detection targets."""
    import SimpleITK as sitk                       # imported lazily

    patients = sorted(
        p.name for p in raw.iterdir()
        if p.is_dir() and p.name.startswith("BraTS")
    )
    if not patients:
        raise RuntimeError(f"no BraTS patient directories under {raw}")

    splits = patient_split(patients, seed=split_seed)
    for name, ids in splits.items():
        d = out / "splits"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}_patients.txt").write_text("\n".join(ids) + "\n")

    rows = []
    for pid in patients:
        vols = []
        for seq in sequences:
            path = raw / pid / f"{pid}_{seq}.nii.gz"
            if not path.exists():
                raise FileNotFoundError(path)
            arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
            # The dataset provides spatially aligned sequences; resample to
            # 1.0 mm isotropic spacing with linear interpolation (D.1).
            vols.append(zscore_nonzero(arr))

        mri = np.stack(vols, axis=-1)              # [Z, H, W, 3]

        seg_path = raw / pid / f"{pid}_seg.nii.gz"
        seg = sitk.GetArrayFromImage(sitk.ReadImage(str(seg_path))).astype(np.int32)
        et = (seg == et_label)                     # enhancing tumor only

        (out / "hr/images" / pid).mkdir(parents=True, exist_ok=True)
        (out / "lr/images" / pid).mkdir(parents=True, exist_ok=True)
        (out / "labels" / pid).mkdir(parents=True, exist_ok=True)

        for z in range(mri.shape[0]):
            if not (mri[z].any()):                 # drop blank slices
                continue

            boxes = np.asarray(
                connected_component_boxes(et[z]), dtype=np.float32
            ).reshape(-1, 4)

            img, boxes_tf, _ = letterbox(mri[z], boxes, size=hr_size)
            yolo = to_yolo(boxes_tf, size=hr_size)

            stem = f"{pid}_z{z:03d}"
            hr_png = out / "hr/images" / pid / f"{stem}.png"
            lr_png = out / "lr/images" / pid / f"{stem}.png"
            lbl = out / "labels" / pid / f"{stem}.txt"

            cv2.imwrite(str(hr_png), (np.clip(img, 0, 1) * 255).astype(np.uint8))
            cv2.imwrite(str(lr_png), (np.clip(make_lr(img, lr_size), 0, 1) * 255).astype(np.uint8))
            lbl.write_text(
                "".join(f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n" for b in yolo)
            )

            rows.append({
                "patient": pid, "z": z, "positive": int(len(yolo) > 0),
                "n_boxes": len(yolo),
                "hr": str(hr_png.relative_to(out)),
                "lr": str(lr_png.relative_to(out)),
                "label": str(lbl.relative_to(out)),
            })

    _write_manifest(out, rows)
    protocol = {
        "dataset": "BraTS2021",
        "modality": "MRI",
        "sequences": list(sequences),
        "target": "ET",
        "target_labels": [et_label],
        "normalization": "per-sequence nonzero z-score; clip [-5,5]; map to [0,1]",
        "hr_size": [hr_size, hr_size],
        "lr_size": [lr_size, lr_size],
        "downsample": "OpenCV INTER_CUBIC",
        "scale": hr_size // lr_size,
        "component_connectivity": 8,
        "split": "patient-level 7:1:2",
        "split_seed": split_seed,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    return protocol


# --------------------------------------------------------------------------- #
# LUNA16 (Appendix D.3)
# --------------------------------------------------------------------------- #
def prepare_luna16(raw: Path, out: Path, split_seed: int,
                   hu_clip=(-1000, 400), hr_size: int = 640,
                   lr_size: int = 160) -> dict:
    """HU-clipped CT slices with slice-wise nodule cross-section boxes (D.3)."""
    import SimpleITK as sitk

    scans = sorted(raw.glob("*.mhd"))
    if not scans:
        raise RuntimeError(f"no .mhd scans under {raw}")

    patients = [s.stem for s in scans]
    splits = patient_split(patients, seed=split_seed)
    (out / "splits").mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        (out / "splits" / f"{name}_patients.txt").write_text("\n".join(ids) + "\n")

    rows = []
    for scan in scans:
        pid = scan.stem
        img = sitk.ReadImage(str(scan))
        vol = sitk.GetArrayFromImage(img).astype(np.float32)          # [Z,H,W]

        origin = np.asarray(img.GetOrigin())[::-1]     # (x,y,z) -> (z,y,x)
        spacing = np.asarray(img.GetSpacing())[::-1]

        vol = np.clip(vol, hu_clip[0], hu_clip[1])
        vol = (vol - hu_clip[0]) / (hu_clip[1] - hu_clip[0])          # -> [0,1]

        nodules = _read_luna16_annotations(raw, pid)                  # (cz,cy,cx,r)

        (out / "hr/images" / pid).mkdir(parents=True, exist_ok=True)
        (out / "lr/images" / pid).mkdir(parents=True, exist_ok=True)
        (out / "labels" / pid).mkdir(parents=True, exist_ok=True)

        for z in range(vol.shape[0]):
            if not vol[z].any():
                continue

            boxes = []
            for cz, cy, cx, r in nodules:
                dz = abs(z - cz) * spacing[0]                          # physical
                if dz > r:                                             # Eq. (D.3)
                    continue
                rs = math.sqrt(max(r * r - dz * dz, 0.0))              # Eq. (D.4)
                ys = rs / spacing[1]
                xs = rs / spacing[2]
                boxes.append([cx - xs, cy - ys, cx + xs, cy + ys])     # Eq. (D.5)

            box_arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
            if len(box_arr):
                h, w = vol[z].shape
                box_arr[:, [0, 2]] = np.clip(box_arr[:, [0, 2]], 0, w)  # clip (D.3)
                box_arr[:, [1, 3]] = np.clip(box_arr[:, [1, 3]], 0, h)

            img3 = np.repeat(vol[z][:, :, None], 3, axis=2)
            img_lb, boxes_tf, _ = letterbox(img3, box_arr, size=hr_size)
            yolo = to_yolo(boxes_tf, size=hr_size)

            stem = f"{pid}_z{z:03d}"
            hr_png = out / "hr/images" / pid / f"{stem}.png"
            lr_png = out / "lr/images" / pid / f"{stem}.png"
            lbl = out / "labels" / pid / f"{stem}.txt"

            cv2.imwrite(str(hr_png), (np.clip(img_lb, 0, 1) * 255).astype(np.uint8))
            cv2.imwrite(str(lr_png), (np.clip(make_lr(img_lb, lr_size), 0, 1) * 255).astype(np.uint8))
            lbl.write_text(
                "".join(f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n" for b in yolo)
            )

            rows.append({
                "patient": pid, "z": z, "positive": int(len(yolo) > 0),
                "n_boxes": len(yolo),
                "hr": str(hr_png.relative_to(out)),
                "lr": str(lr_png.relative_to(out)),
                "label": str(lbl.relative_to(out)),
            })

    _write_manifest(out, rows)
    protocol = {
        "dataset": "LUNA16",
        "modality": "CT",
        "hu_clip": list(hu_clip),
        "target": "slice-wise nodule cross-section",
        "hr_size": [hr_size, hr_size],
        "lr_size": [lr_size, lr_size],
        "downsample": "OpenCV INTER_CUBIC",
        "scale": hr_size // lr_size,
        "split": "patient-level 7:1:2",
        "split_seed": split_seed,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    return protocol


def _read_luna16_annotations(raw: Path, pid: str):
    """Parse LUNA16 annotations (physical coords) into voxel (cz, cy, cx, r)."""
    import SimpleITK as sitk

    cand = raw / "annotations.csv"
    if not cand.exists():
        return []

    img = sitk.ReadImage(str(raw / f"{pid}.mhd"))
    origin = np.asarray(img.GetOrigin())       # (x, y, z)
    spacing = np.asarray(img.GetSpacing())
    inv = 1.0 / spacing

    out = []
    with cand.open() as fh:
        for row in csv.DictReader(fh):
            if row["seriesuid"] != pid:
                continue
            world = np.asarray([float(row["coordX"]), float(row["coordY"]),
                                float(row["coordZ"])])
            vox = (world - origin) * inv        # (x, y, z) voxel
            cx, cy, cz = vox[0], vox[1], vox[2]
            r = float(row["diameter_mm"]) / 2.0
            out.append((cz, cy, cx, r))
    return out


# --------------------------------------------------------------------------- #
# VinDr-CXR (Appendix D.4)
# --------------------------------------------------------------------------- #
def prepare_vindrcxr(raw: Path, out: Path, split_seed: int,
                     percentiles=(0.5, 99.5), hr_size: int = 640,
                     lr_size: int = 160) -> dict:
    """Percentile-clipped radiographs replicated to three channels (D.4)."""
    images = sorted(list(raw.glob("**/*.dicom")) + list(raw.glob("**/*.dcm")))
    if not images:
        raise RuntimeError(f"no DICOM files under {raw}")

    patients = [p.stem for p in images]
    splits = patient_split(patients, seed=split_seed)
    (out / "splits").mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        (out / "splits" / f"{name}_patients.txt").write_text("\n".join(ids) + "\n")

    rows = []
    for path in images:
        pid = path.stem
        gray = _read_dicom_gray(path)
        lo, hi = np.percentile(gray, percentiles)
        gray = np.clip((gray - lo) / max(hi - lo, 1e-8), 0.0, 1.0).astype(np.float32)
        img3 = np.repeat(gray[:, :, None], 3, axis=2)

        boxes = _read_vindr_boxes(raw, pid, gray.shape)
        img_lb, boxes_tf, _ = letterbox(
            img3, np.asarray(boxes, dtype=np.float32).reshape(-1, 4), size=hr_size
        )
        yolo = to_yolo(boxes_tf, size=hr_size)

        (out / "hr/images").mkdir(parents=True, exist_ok=True)
        (out / "lr/images").mkdir(parents=True, exist_ok=True)
        (out / "labels").mkdir(parents=True, exist_ok=True)

        hr_png = out / "hr/images" / f"{pid}.png"
        lr_png = out / "lr/images" / f"{pid}.png"
        lbl = out / "labels" / f"{pid}.txt"

        cv2.imwrite(str(hr_png), (img_lb * 255).astype(np.uint8))
        cv2.imwrite(str(lr_png), (make_lr(img_lb, lr_size) * 255).astype(np.uint8))
        lbl.write_text(
            "".join(f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n" for b in yolo)
        )

        rows.append({
            "patient": pid, "z": 0, "positive": int(len(yolo) > 0),
            "n_boxes": len(yolo),
            "hr": str(hr_png.relative_to(out)),
            "lr": str(lr_png.relative_to(out)),
            "label": str(lbl.relative_to(out)),
        })

    _write_manifest(out, rows)
    protocol = {
        "dataset": "VinDr-CXR",
        "modality": "X-ray",
        "percentile_clip": list(percentiles),
        "channels": 3,
        "target": "native 2D boxes transformed with identical letterbox parameters",
        "hr_size": [hr_size, hr_size],
        "lr_size": [lr_size, lr_size],
        "downsample": "OpenCV INTER_CUBIC",
        "scale": hr_size // lr_size,
        "split": "patient-level 7:1:2",
        "split_seed": split_seed,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    return protocol


def _read_dicom_gray(path: Path) -> np.ndarray:
    import pydicom
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    return arr


def _read_vindr_boxes(raw: Path, pid: str, shape):
    """Native VinDr-CXR boxes; override for a different annotation layout."""
    csv_path = raw / "annotations" / "train.csv"
    if not csv_path.exists():
        return []
    boxes = []
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("image_id") != pid:
                continue
            boxes.append([float(row["x_min"]), float(row["y_min"]),
                          float(row["x_max"]), float(row["y_max"])])
    return boxes


# --------------------------------------------------------------------------- #
def _write_manifest(out: Path, rows: list) -> None:
    if not rows:
        return
    with (out / "manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True,
                    choices=["brats2021", "luna16", "vindrcxr"])
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--hr-size", type=int, default=640)
    ap.add_argument("--lr-size", type=int, default=160)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if args.dataset == "brats2021":
        p = prepare_brats2021(args.raw, args.out, args.split_seed,
                              hr_size=args.hr_size, lr_size=args.lr_size)
    elif args.dataset == "luna16":
        p = prepare_luna16(args.raw, args.out, args.split_seed,
                           hr_size=args.hr_size, lr_size=args.lr_size)
    else:
        p = prepare_vindrcxr(args.raw, args.out, args.split_seed,
                             hr_size=args.hr_size, lr_size=args.lr_size)

    print(json.dumps(p, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
