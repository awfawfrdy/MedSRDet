# VinDr-CXR split — regenerable, not redistributed

**Status:** the split files `train_patients.txt`, `val_patients.txt`,
`test_patients.txt` are **NOT shipped in this repository**. They are regenerated
locally from the official release; the exact procedure is below.

## Why this directory has no manifest file
- VinDr-CXR 1.0.0 is distributed by PhysioNet under **Credentialed Access**
  (a Data Use Agreement must be signed). Redistributing the split manifest
  would violate that DUA, so we ship the *generator* and the *procedure*
  instead of the derived file.
- The public release exposes only anonymised `image_id` (a hash of the DICOM
  SOP Instance UID). It does **not** expose `patient_id`. Consequently this
  dataset's partition is **image-level**, whereas `brats2021/` and `luna16/`
  above are **patient-level**. This is a property of the dataset, not a choice.

## Reproduce the exact split
```bash
# 1. Credentialed PhysioNet access to vindr-cxr 1.0.0 + a Personal Access Token
# 2. Download the annotation CSVs (images are NOT needed for the split)
python scripts/download_vindr.py --token $PHYSIO_TOKEN --out ./vindr-cxr-1.0.0

# 3. Generate the fixed 7:1:2 partition (image-level, split seed 42)
python scripts/gen_vindr_manifest.py \
    --annotations ./vindr-cxr-1.0.0/annotations_train.csv \
                ./vindr-cxr-1.0.0/annotations_test.csv
#    -> writes data/splits/vindr/{train,val,test}_patients.txt
```

## Split specification (identical across all three datasets)
- Ratios: **70 % / 10 % / 20 %** (train / val / test)
- `SPLIT_SEED = 42`, independent of the optimisation seeds 42–46
- Deterministic, leakage-free, union-complete (validated inside
  `scripts/gen_vindr_manifest.py`)

## Note on the official release split
The official VinDr-CXR release ships a fixed ~15k / ~3k train/test partition
(~83 / 17). The 7:1:2 partition used in this work is an independent re-split
over the same pool of image ids, produced by the script above so that all three
datasets share one fixed partition scheme.
