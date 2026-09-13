"""Generate the VinDr-CXR split manifest.

VinDr-CXR (PhysioNet project ``vindr-cxr`` 1.0.0, Credentialed Access) releases
each chest X-ray as an independent, anonymised *image* whose ID is a hash of the
DICOM SOP Instance UID.  The public annotations (``annotations_train.csv`` /
``annotations_test.csv``) expose only ``image_id``; they do NOT contain a
patient identifier.  Consequently a strictly *patient-level* split is impossible
from the released data.

This script therefore supports two modes:

  * IMAGE-LEVEL (default): partition on ``image_id`` -- the only unit VinDr
    releases.  Honest for VinDr, but the manuscript's "patient-level" claim then
    applies only to BraTS2021 + LUNA16, not to V.
  * PATIENT-LEVEL (optional): if a PatientID-per-image mapping can be recovered
    (e.g. from the downloaded DICOM tags before de-identification, or any
    mapping you obtain), pass ``--patient-map image_id,patient_id`` and the split
    becomes truly patient-level.

Either way the partition uses the SAME deterministic algorithm and the
INDEPENDENT split seed (``common.split_seed`` = 42), decoupled from the
optimisation seeds 42-46, and is frozen once written.

The official VinDr release ships a fixed 15k/3k train/test split (~83/17), which
differs from our 7:1:2 ratio.  This script produces OUR 7:1:2 ratio for ratio
uniformity across the three datasets; if you prefer to honour VinDr's official
split instead, do NOT use this script and document the official split directly.

Usage
-----
    # image-level (default) -- derive ids from the official annotation CSVs
    python scripts/gen_vindr_manifest.py --annotations annotations_train.csv annotations_test.csv
    # or from a plain newline list of image_ids
    python scripts/gen_vindr_manifest.py --image-ids vindr_all_image_ids.txt
    # patient-level (only if a mapping is available)
    python scripts/gen_vindr_manifest.py --image-ids vindr_all_image_ids.txt --patient-map map.csv
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SPLIT_DIR = REPO / "data" / "splits" / "vindr"
SPLIT_SEED = 42
RATIOS = (0.7, 0.1, 0.2)


def patient_split(unit_ids, ratios=RATIOS, seed=SPLIT_SEED):
    """Identical algorithm to scripts/preprocess.patient_split."""
    ids = sorted(unit_ids)
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


def load_image_ids(args):
    if args.image_ids:
        return [l.strip() for l in args.image_ids.read_text().splitlines() if l.strip()]
    ids: set[str] = set()
    for csvp in args.annotations:
        with csvp.open() as fh:
            r = csv.DictReader(fh)
            col = "image_id" if "image_id" in (r.fieldnames or []) else r.fieldnames[0]
            for row in r:
                ids.add(row[col].strip())
    return sorted(ids)


def load_patient_map(path: Path):
    m: dict[str, str] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            m[row["image_id"].strip()] = row["patient_id"].strip()
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", nargs="+", type=Path, default=[],
                    help="official annotations_train.csv / annotations_test.csv "
                         "(image_id extracted from the 'image_id' column)")
    ap.add_argument("--image-ids", type=Path, default=None,
                    help="newline list of all VinDr image_ids")
    ap.add_argument("--patient-map", type=Path, default=None,
                    help="optional CSV with columns image_id,patient_id for a "
                         "truly patient-level split")
    args = ap.parse_args()

    if not args.annotations and not args.image_ids:
        raise SystemExit("provide --annotations and/or --image-ids")

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    image_ids = load_image_ids(args)
    if not image_ids:
        raise SystemExit("no image_ids resolved")

    if args.patient_map:
        pmap = load_patient_map(args.patient_map)
        missing = [i for i in image_ids if i not in pmap]
        if missing:
            raise SystemExit(f"{len(missing)} image_ids lack a patient mapping "
                             f"(e.g. {missing[:3]})")
        unit_to_imgs: dict[str, list[str]] = {}
        for i in image_ids:
            unit_to_imgs.setdefault(pmap[i], []).append(i)
        units = sorted(unit_to_imgs)
        level = "patient"
    else:
        unit_to_imgs = {i: [i] for i in image_ids}
        units = image_ids
        level = "image"
        print("[gen][WARN] no --patient-map supplied -> IMAGE-LEVEL split. "
              "VinDr-CXR releases no patient ID, so this is the honest unit. "
              "The manuscript 'patient-level' claim then covers BraTS2021 + "
              "LUNA16 only, NOT V.")

    splits = patient_split(units, seed=SPLIT_SEED)

    stats: dict[str, tuple[int, int]] = {}
    all_ids: set[str] = set()
    for name, us in splits.items():
        ids: list[str] = []
        for u in us:
            ids.extend(sorted(unit_to_imgs[u]))
        all_ids |= set(ids)
        (SPLIT_DIR / f"{name}_patients.txt").write_text("\n".join(sorted(ids)) + "\n")
        stats[name] = (len(us), len(ids))

    # ---- validation ----
    usets = {k: set(v) for k, v in splits.items()}
    for a in usets:
        for b in usets:
            if a < b:
                leak = usets[a] & usets[b]
                assert not leak, f"leakage between {a} and {b}: {leak}"
    assert all_ids == set(image_ids), "manifest union != full image_id set"
    if args.patient_map:
        dual = [u for u, ss in unit_to_imgs.items() if len(ss) > 1]
        for u in dual:
            infold = sum(u in set(p) for p in splits.values())
            assert infold == 1, f"multi-image unit {u} split across folds"
        print(f"[gen] multi-image patients kept together: {dual}")

    print(f"[gen] split level        = {level}")
    print(f"[gen] {level}-level counts : "
          f"train={stats['train'][0]} val={stats['val'][0]} test={stats['test'][0]}")
    print(f"[gen] image-level counts  : "
          f"train={stats['train'][1]} val={stats['val'][1]} test={stats['test'][1]}")
    print("[gen] VALIDATION OK: no leakage, union complete"
          + (", multi-image patients intact" if args.patient_map else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
