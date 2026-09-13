"""Dataset preprocessing and 2D detection-target construction.

Implements Section 4.2 and Appendix D (Tables D.1) of the manuscript:

  D.1  common protocol         : 1.0 mm isotropic resampling (CT/MRI, image =
                                 linear interpolation, segmentation mask =
                                 nearest-neighbour), letterbox to 640x640,
                                 intensities in [0, 1], LR target obtained by
                                 bicubic downsampling to 160x160.
  D.2  BraTS2021 (MRI)         : T1ce / T2 / FLAIR stacked as three channels,
                                 per-sequence z-score inside the non-zero brain
                                 region, clipped to [-5, 5], mapped to [0, 1];
                                 enhancing-tumor (ET, label 4) mask converted
                                 to axial connected components -> tight boxes.
  D.3  LUNA16 (CT)             : HU clipped to [-1000, 400] and mapped to
                                 [0, 1]; nodule centre/diameter converted to
                                 slice-wise cross-section boxes with
                                 r_s = sqrt(r^2 - dz^2).  Nodule annotations
                                 are physical coordinates and are converted
                                 with ``TransformPhysicalPointToContinuousIndex``
                                 in the RESAMPLED image geometry.
  D.4  VinDr-CXR (X-ray)       : 0.5-99.5 percentile clipping, mapped to
                                 [0, 1], replicated to three channels; native
                                 2D boxes (with their NATIVE abnormality class
                                 ids, 14-class label space) transformed with
                                 identical letterbox parameters.

Patient-level train/val/test partitioning is performed BEFORE slice extraction
so that no patient contributes to more than one subset.  Frozen split
manifests (``--split-manifest-root``) take priority over re-randomised
splits: if ``<root>/<dataset>/{train,val,test}_patients.txt`` exist they are
used verbatim.

Usage
-----
    python scripts/preprocess.py --dataset brats2021 \
        --raw /path/to/BraTS2021 --out /path/to/prepared --split-seed 42 \
        --split-manifest-root data/splits

Output layout (per dataset)::

    hr/images/<patient>/<patient>_z<zzz>.png     640x640
    lr/images/<patient>/<patient>_z<zzz>.png     160x160
    labels/<patient>/<patient>_z<zzz>.txt        YOLO class_id cx cy w h
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


def to_yolo(boxes: np.ndarray, size: int = 640, class_ids=None) -> list:
    """Convert (x1, y1, x2, y2) to normalised ``class_id cx cy w h`` rows.

    ``class_ids`` aligns with ``boxes``; when omitted every row gets class 0
    (the single-class label space of BraTS2021 / LUNA16).
    """
    out = []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cx = ((x1 + x2) / 2.0) / size
        cy = ((y1 + y2) / 2.0) / size
        bw = (x2 - x1) / size
        bh = (y2 - y1) / size
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        bw, bh = min(max(bw, 0.0), 1.0), min(max(bh, 0.0), 1.0)
        if bw <= 0 or bh <= 0:
            continue
        cid = 0 if class_ids is None else int(class_ids[i])
        out.append([cid, cx, cy, bw, bh])
    return out


def write_yolo_labels(path: Path, rows: list) -> None:
    path.write_text(
        "".join(f"{r[0]} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}\n"
                for r in rows)
    )


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


def load_split_manifest(root: Path | None, dataset: str):
    """Load a frozen split manifest if available, else return None.

    Expected layout: ``<root>/<dataset>/{train,val,test}_patients.txt``.
    """
    if root is None:
        return None
    d = Path(root) / dataset
    files = {s: d / f"{s}_patients.txt" for s in ("train", "val", "test")}
    if not all(f.exists() for f in files.values()):
        return None
    return {
        s: {line.strip() for line in f.read_text().splitlines() if line.strip()}
        for s, f in files.items()
    }


def resolve_splits(patient_ids, manifest_root: Path | None, dataset: str,
                   split_seed: int):
    """Frozen manifest when available (P0), otherwise random 7:1:2 split."""
    frozen = load_split_manifest(manifest_root, dataset)
    if frozen is not None:
        print(f"[splits] using frozen manifest {Path(manifest_root) / dataset}")
        return frozen
    print(f"[splits] no frozen manifest for {dataset!r}; "
          f"generating random patient-level 7:1:2 split (seed {split_seed})")
    return patient_split(patient_ids, seed=split_seed)


def write_split_files(out: Path, splits: dict) -> None:
    (out / "splits").mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        (out / "splits" / f"{name}_patients.txt").write_text(
            "\n".join(sorted(ids)) + "\n")


# --------------------------------------------------------------------------- #
# 1 mm isotropic resampling (Appendix D.1) — SimpleITK
# --------------------------------------------------------------------------- #
def resample_to_isotropic(img, spacing=(1.0, 1.0, 1.0),
                          interpolator=None):
    """Resample a SimpleITK image to isotropic voxel spacing.

    Args:
        img: sitk.Image.
        spacing: target voxel spacing in mm (default 1 mm isotropic).
        interpolator: sitk interpolator; defaults to linear (images).  Use
            ``sitk.sitkNearestNeighbor`` for segmentation masks (D.1).
    """
    import SimpleITK as sitk

    if interpolator is None:
        interpolator = sitk.sitkLinear

    old_spacing = np.asarray(img.GetSpacing(), dtype=np.float64)
    old_size = np.asarray(img.GetSize(), dtype=np.float64)
    new_size = [int(round(s * sp / n)) for s, sp, n in
                zip(old_size, old_spacing, spacing)]

    f = sitk.ResampleImageFilter()
    f.SetOutputSpacing(tuple(float(v) for v in spacing))
    f.SetSize([int(v) for v in new_size])
    f.SetOutputOrigin(img.GetOrigin())
    f.SetOutputDirection(img.GetDirection())
    f.SetTransform(sitk.Transform())
    f.SetInterpolator(interpolator)
    return f.Execute(img)


def resample_onto_reference(img, ref, interpolator=None):
    """Resample ``img`` onto the exact grid of the reference image."""
    import SimpleITK as sitk

    if interpolator is None:
        interpolator = sitk.sitkLinear
    f = sitk.ResampleImageFilter()
    f.SetReferenceImage(ref)
    f.SetInterpolator(interpolator)
    f.SetTransform(sitk.Transform())
    return f.Execute(img)


# --------------------------------------------------------------------------- #
# BraTS2021 (Appendix D.2)
# --------------------------------------------------------------------------- #
def prepare_brats2021(raw: Path, out: Path, split_seed: int,
                      sequences=("t1ce", "t2", "flair"),
                      et_label: int = 4, hr_size: int = 640,
                      lr_size: int = 160,
                      manifest_root: Path | None = None) -> dict:
    """Build the three-channel MRI input and ET detection targets.

    All sequences AND the segmentation are resampled to 1.0 mm isotropic
    spacing (D.1): images with linear interpolation, the segmentation mask
    with nearest-neighbour interpolation, all on the SAME reference grid.
    """
    import SimpleITK as sitk

    patients = sorted(
        p.name for p in raw.iterdir()
        if p.is_dir() and p.name.startswith("BraTS")
    )
    if not patients:
        raise RuntimeError(f"no BraTS patient directories under {raw}")

    write_split_files(out, resolve_splits(patients, manifest_root,
                                          "brats2021", split_seed))
    splits = load_split_manifest(manifest_root, "brats2021") or \
        patient_split(patients, seed=split_seed)

    rows = []
    for pid in patients:
        seq_imgs = []
        for seq in sequences:
            path = raw / pid / f"{pid}_{seq}.nii.gz"
            if not path.exists():
                raise FileNotFoundError(path)
            seq_imgs.append(sitk.ReadImage(str(path)))

        # Reference grid = first sequence resampled to 1 mm isotropic (linear).
        ref = resample_to_isotropic(seq_imgs[0], (1.0, 1.0, 1.0),
                                    sitk.sitkLinear)
        vols = []
        for img in seq_imgs:
            img_1mm = resample_onto_reference(img, ref, sitk.sitkLinear)
            arr = sitk.GetArrayFromImage(img_1mm).astype(np.float32)
            vols.append(zscore_nonzero(arr))       # D.2 normalisation

        mri = np.stack(vols, axis=-1)              # [Z, H, W, 3]

        seg_path = raw / pid / f"{pid}_seg.nii.gz"
        seg_img = sitk.ReadImage(str(seg_path))
        seg_1mm = resample_onto_reference(seg_img, ref,
                                          sitk.sitkNearestNeighbor)  # D.1 mask
        seg = sitk.GetArrayFromImage(seg_1mm).astype(np.int32)
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
            yolo = to_yolo(boxes_tf, size=hr_size)          # single class 0

            stem = f"{pid}_z{z:03d}"
            hr_png = out / "hr/images" / pid / f"{stem}.png"
            lr_png = out / "lr/images" / pid / f"{stem}.png"
            lbl = out / "labels" / pid / f"{stem}.txt"

            cv2.imwrite(str(hr_png), (np.clip(img, 0, 1) * 255).astype(np.uint8))
            cv2.imwrite(str(lr_png), (np.clip(make_lr(img, lr_size), 0, 1) * 255).astype(np.uint8))
            write_yolo_labels(lbl, yolo)

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
        "resampling": "1.0 mm isotropic; image linear, mask nearest-neighbour; "
                      "shared reference grid across sequences and mask",
        "normalization": "per-sequence nonzero z-score; clip [-5,5]; map to [0,1]",
        "hr_size": [hr_size, hr_size],
        "lr_size": [lr_size, lr_size],
        "downsample": "OpenCV INTER_CUBIC",
        "scale": hr_size // lr_size,
        "component_connectivity": 8,
        "num_classes": 1,
        "split": "patient-level 7:1:2 (frozen manifest when available)",
        "split_seed": split_seed,
        "split_source": "manifest" if load_split_manifest(manifest_root, "brats2021") else "random",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    return protocol


# --------------------------------------------------------------------------- #
# LUNA16 (Appendix D.3)
# --------------------------------------------------------------------------- #
def prepare_luna16(raw: Path, out: Path, split_seed: int,
                   hu_clip=(-1000, 400), hr_size: int = 640,
                   lr_size: int = 160,
                   manifest_root: Path | None = None) -> dict:
    """HU-clipped CT slices with slice-wise nodule cross-section boxes (D.3).

    The CT volume is resampled to 1.0 mm isotropic spacing with linear
    interpolation BEFORE slice extraction; nodule annotations are physical
    coordinates converted in the RESAMPLED image geometry.
    """
    import SimpleITK as sitk

    scans = sorted(raw.glob("*.mhd"))
    if not scans:
        raise RuntimeError(f"no .mhd scans under {raw}")

    patients = [s.stem for s in scans]
    write_split_files(out, resolve_splits(patients, manifest_root,
                                          "luna16", split_seed))
    splits = load_split_manifest(manifest_root, "luna16") or \
        patient_split(patients, seed=split_seed)

    rows = []
    for scan in scans:
        pid = scan.stem
        img = sitk.ReadImage(str(scan))

        # ---- D.1: 1 mm isotropic resampling (linear interpolation) ----
        img_1mm = resample_to_isotropic(img, (1.0, 1.0, 1.0), sitk.sitkLinear)
        vol = sitk.GetArrayFromImage(img_1mm).astype(np.float32)      # [Z,H,W]
        spacing_xyz = np.asarray(img_1mm.GetSpacing(), dtype=np.float64)  # (x,y,z) mm

        vol = np.clip(vol, hu_clip[0], hu_clip[1])
        vol = (vol - hu_clip[0]) / (hu_clip[1] - hu_clip[0])          # -> [0,1]

        # Nodule physical coords -> continuous indices in the RESAMPLED grid.
        nodules = _read_luna16_annotations(raw, pid, img_1mm)  # (cz,cy,cx,r_vox)

        (out / "hr/images" / pid).mkdir(parents=True, exist_ok=True)
        (out / "lr/images" / pid).mkdir(parents=True, exist_ok=True)
        (out / "labels" / pid).mkdir(parents=True, exist_ok=True)

        for z in range(vol.shape[0]):
            if not vol[z].any():
                continue

            boxes = []
            for cz, cy, cx, r_vox in nodules:
                dz = abs(z - cz) * spacing_xyz[2]                     # physical mm
                if dz > r_vox * spacing_xyz[2]:
                    continue
                rs = math.sqrt(max((r_vox * spacing_xyz[2]) ** 2 - dz * dz, 0.0))
                ys = rs / spacing_xyz[1]                              # Eq. (D.4)
                xs = rs / spacing_xyz[0]
                boxes.append([cx - xs, cy - ys, cx + xs, cy + ys])    # Eq. (D.5)

            box_arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
            if len(box_arr):
                h, w = vol[z].shape
                box_arr[:, [0, 2]] = np.clip(box_arr[:, [0, 2]], 0, w)  # clip (D.3)
                box_arr[:, [1, 3]] = np.clip(box_arr[:, [1, 3]], 0, h)

            img3 = np.repeat(vol[z][:, :, None], 3, axis=2)
            img_lb, boxes_tf, _ = letterbox(img3, box_arr, size=hr_size)
            yolo = to_yolo(boxes_tf, size=hr_size)           # single class 0

            stem = f"{pid}_z{z:03d}"
            hr_png = out / "hr/images" / pid / f"{stem}.png"
            lr_png = out / "lr/images" / pid / f"{stem}.png"
            lbl = out / "labels" / pid / f"{stem}.txt"

            cv2.imwrite(str(hr_png), (np.clip(img_lb, 0, 1) * 255).astype(np.uint8))
            cv2.imwrite(str(lr_png), (np.clip(make_lr(img_lb, lr_size), 0, 1) * 255).astype(np.uint8))
            write_yolo_labels(lbl, yolo)

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
        "resampling": "1.0 mm isotropic; linear interpolation; nodule physical "
                      "coordinates converted in the resampled geometry "
                      "(TransformPhysicalPointToContinuousIndex)",
        "target": "slice-wise nodule cross-section",
        "hr_size": [hr_size, hr_size],
        "lr_size": [lr_size, lr_size],
        "downsample": "OpenCV INTER_CUBIC",
        "scale": hr_size // lr_size,
        "num_classes": 1,
        "split": "patient-level 7:1:2 (frozen manifest when available)",
        "split_seed": split_seed,
        "split_source": "manifest" if load_split_manifest(manifest_root, "luna16") else "random",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    return protocol


def _read_luna16_annotations(raw: Path, pid: str, img_1mm):
    """Parse LUNA16 annotations (physical coords) in the RESAMPLED geometry.

    Returns list of (cz, cy, cx, r_voxels): continuous voxel indices of the
    nodule centre in the 1 mm grid and the radius in voxels.
    """
    import SimpleITK as sitk

    cand = raw / "annotations.csv"
    if not cand.exists():
        return []

    out = []
    with cand.open() as fh:
        for row in csv.DictReader(fh):
            if row["seriesuid"] != pid:
                continue
            world = (float(row["coordX"]), float(row["coordY"]),
                     float(row["coordZ"]))
            # Physical point -> continuous index in the resampled geometry.
            idx = img_1mm.TransformPhysicalPointToContinuousIndex(world)
            cx, cy, cz = idx[0], idx[1], idx[2]
            spacing = img_1mm.GetSpacing()[0]        # isotropic -> mm per voxel
            r = float(row["diameter_mm"]) / 2.0 / spacing
            out.append((cz, cy, cx, r))
    return out


# --------------------------------------------------------------------------- #
# VinDr-CXR (Appendix D.4) — native 14-class label space
# --------------------------------------------------------------------------- #
VINDR_NUM_CLASSES = 14          # manuscript: dataset-specific 14-class space
VINDR_EXCLUDED = {"No Finding"}  # image-level "no abnormality" rows


def build_vindr_class_map(rows: list) -> dict:
    """Fixed, reproducible mapping class_name -> class_id (1..14).

    Preference order:
      1. the official ``class_id`` column (0 = "No Finding" is excluded);
      2. otherwise alphabetical assignment 1..N over the abnormality names.
    Raises if more than 14 abnormality classes appear.
    """
    names = sorted({r["class_name"] for r in rows
                    if r["class_name"] not in VINDR_EXCLUDED})
    if len(names) > VINDR_NUM_CLASSES:
        raise ValueError(
            f"VinDr-CXR mapping exceeds {VINDR_NUM_CLASSES} classes: {names}"
        )

    cmap: dict = {}
    if rows and "class_id" in rows[0]:
        for r in rows:
            name = r["class_name"]
            if name in VINDR_EXCLUDED:
                continue
            cid = int(r["class_id"])
            if name in cmap and cmap[name] != cid:
                raise ValueError(f"inconsistent class_id for {name!r}")
            cmap[name] = cid
    else:
        cmap = {name: i + 1 for i, name in enumerate(names)}

    for name, cid in cmap.items():
        if not (0 <= cid < VINDR_NUM_CLASSES):
            raise ValueError(f"class id {cid} out of 14-class range for {name!r}")
    return cmap


def prepare_vindrcxr(raw: Path, out: Path, split_seed: int,
                     percentiles=(0.5, 99.5), hr_size: int = 640,
                     lr_size: int = 160,
                     manifest_root: Path | None = None) -> dict:
    """Percentile-clipped radiographs replicated to three channels (D.4).

    Labels keep the NATIVE abnormality class id of every box (14-class label
    space); they are never collapsed to class 0.
    """
    images = sorted(list(raw.glob("**/*.dicom")) + list(raw.glob("**/*.dcm")))
    if not images:
        raise RuntimeError(f"no DICOM files under {raw}")

    patients = [p.stem for p in images]
    write_split_files(out, resolve_splits(patients, manifest_root,
                                          "vindrcxr", split_seed))
    splits = load_split_manifest(manifest_root, "vindrcxr") or \
        patient_split(patients, seed=split_seed)

    annotations, class_map = _read_vindr_annotations(raw)

    rows = []
    for path in images:
        pid = path.stem
        gray = _read_dicom_gray(path)
        lo, hi = np.percentile(gray, percentiles)
        gray = np.clip((gray - lo) / max(hi - lo, 1e-8), 0.0, 1.0).astype(np.float32)
        img3 = np.repeat(gray[:, :, None], 3, axis=2)

        ann = annotations.get(pid, [])
        boxes = np.asarray([a[1:] for a in ann], dtype=np.float32).reshape(-1, 4)
        class_ids = [a[0] for a in ann]

        img_lb, boxes_tf, _ = letterbox(
            img3, boxes, size=hr_size
        )
        yolo = to_yolo(boxes_tf, size=hr_size, class_ids=class_ids)

        (out / "hr/images").mkdir(parents=True, exist_ok=True)
        (out / "lr/images").mkdir(parents=True, exist_ok=True)
        (out / "labels").mkdir(parents=True, exist_ok=True)

        hr_png = out / "hr/images" / f"{pid}.png"
        lr_png = out / "lr/images" / f"{pid}.png"
        lbl = out / "labels" / f"{pid}.txt"

        cv2.imwrite(str(hr_png), (img_lb * 255).astype(np.uint8))
        cv2.imwrite(str(lr_png), (make_lr(img_lb, lr_size) * 255).astype(np.uint8))
        write_yolo_labels(lbl, yolo)

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
        "num_classes": VINDR_NUM_CLASSES,
        "class_map": class_map,
        "label_note": "native abnormality class ids preserved per box; "
                      "'No Finding' rows excluded",
        "target": "native 2D boxes transformed with identical letterbox parameters",
        "hr_size": [hr_size, hr_size],
        "lr_size": [lr_size, lr_size],
        "downsample": "OpenCV INTER_CUBIC",
        "scale": hr_size // lr_size,
        "split": "patient-level 7:1:2 (frozen manifest when available)",
        "split_seed": split_seed,
        "split_source": "manifest" if load_split_manifest(manifest_root, "vindrcxr") else "random",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    return protocol


def _read_dicom_gray(path: Path) -> np.ndarray:
    import pydicom
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    return arr


def _read_vindr_annotations(raw: Path):
    """Parse the official VinDr-CXR annotation CSV.

    Returns (annotations, class_map):
      annotations: {image_id: [(class_id, x_min, y_min, x_max, y_max), ...]}
      class_map:   {class_name: class_id} (fixed, reproducible)
    """
    annotations: dict = {}
    rows = []
    csv_path = raw / "annotations" / "train.csv"
    if not csv_path.exists():
        # fall back to any CSV under the raw root
        candidates = sorted(raw.glob("*.csv"))
        csv_path = candidates[0] if candidates else None
    if csv_path is None or not csv_path.exists():
        return annotations, {}

    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
            if row.get("class_name") in VINDR_EXCLUDED:
                continue
            if "class_id" not in row:
                continue  # handled after the class map is built
            cid = int(row["class_id"])
            annotations.setdefault(row["image_id"], []).append(
                (cid, float(row["x_min"]), float(row["y_min"]),
                 float(row["x_max"]), float(row["y_max"]))
            )

    class_map = build_vindr_class_map(rows)

    # If the CSV lacked class_id columns, remap by class_name.
    if rows and "class_id" not in rows[0]:
        annotations = {}
        for row in rows:
            name = row.get("class_name")
            if name in VINDR_EXCLUDED or name is None:
                continue
            annotations.setdefault(row["image_id"], []).append(
                (class_map[name], float(row["x_min"]), float(row["y_min"]),
                 float(row["x_max"]), float(row["y_max"]))
            )
    return annotations, class_map


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
    ap.add_argument("--split-manifest-root", type=Path, default=None,
                    help="root of frozen split manifests "
                         "(e.g. data/splits); overrides random splits")
    ap.add_argument("--hr-size", type=int, default=640)
    ap.add_argument("--lr-size", type=int, default=160)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if args.dataset == "brats2021":
        p = prepare_brats2021(args.raw, args.out, args.split_seed,
                              hr_size=args.hr_size, lr_size=args.lr_size,
                              manifest_root=args.split_manifest_root)
    elif args.dataset == "luna16":
        p = prepare_luna16(args.raw, args.out, args.split_seed,
                           hr_size=args.hr_size, lr_size=args.lr_size,
                           manifest_root=args.split_manifest_root)
    else:
        p = prepare_vindrcxr(args.raw, args.out, args.split_seed,
                             hr_size=args.hr_size, lr_size=args.lr_size,
                             manifest_root=args.split_manifest_root)

    print(json.dumps(p, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
