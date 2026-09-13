# MedSRDet — Anonymous Reviewer Repository

Implementation of **"MedSRDet: A Unified Framework of Frequency-Aware
Super-Resolution and Modality-Robust Representation Regularization for Small
Lesion Detection"** (Biomedical Signal Processing and Control, under review).

---

## 1. Environment

Ubuntu 20.04 LTS, Python 3.8.12, PyTorch 2.2.0, CUDA 11.3 + cuDNN 8.2
(manuscript Table 5); trained on NVIDIA RTX 4090 (24 GB) and
8×A100-SXM4-80GB.

```bash
pip install -r requirements.txt
```

The YOLOv12n shared feature encoder is loaded from the installed
`ultralytics` package (checkpoint resolved automatically on first use).

---

## 2. Repository layout

```
MedSRDet/
├── modules/
│   ├── mfasrnet_arch.py     # MFASR network (RRDB trunk, ×4)
│   ├── rrdbnet_arch.py      # RRDB / ResidualDenseBlock backbone
│   ├── loss_mfasr.py        # L_SR = L_pix + 0.1 L_freq + 0.05 L_perc
│   ├── cmcl.py              # CMCL contrastive regularizer (training only)
│   ├── medhead.py           # MedHead detection head
│   ├── yolo_encoder.py      # shared YOLOv12n P3/P4/P5 encoder
│   ├── detection_utils.py   # decoding, NMS, mAP@0.5
│   └── _arch_utils.py       # BasicSR-compatible helpers (Apache-2.0)
├── scripts/
│   ├── preprocess.py        # preprocessing + 2D target construction
│   ├── lesion_views.py      # CMCL lesion-view construction
│   ├── train.py             # joint optimisation
│   ├── evaluate.py          # mAP@0.5 evaluation
│   ├── inference.py         # deployment pathway
│   ├── selfcheck.py         # configuration checks
│   └── smoke_test.py        # synthetic-data pipeline check
├── configs/
│   └── default.yaml         # experiment configuration
├── data/splits/brats2021/   # patient-level split manifest
├── requirements.txt
├── LICENSE
└── LICENSE_BasicSR.txt      # Apache-2.0, covers the BasicSR-derived code
```

---

## 3. Modules

**MFASR** (Methods 3.2, Table 2) — RRDB-based reconstruction backbone:
`3 → 64` shallow convolution, 23 RRDB blocks (3 RDBs each, growth channels
32, local residual scaling 0.2), two-stage nearest-neighbour upsampling
(overall ×4).  Objective (Eq. 1): `L_SR = L_pix + 0.1·L_freq + 0.05·L_perc`
with L1 on the FFT magnitude spectrum and frozen VGG-19 ReLU5_4 features.

**CMCL** (Methods 3.3, Table 3, Appendix B/C) — for every annotated lesion
instance, an augmented view and an MFASR-enhanced view of the same
lesion-centred support form the positive pair; negatives are the other
in-batch instances plus the momentum memory queue (K = 65,536).  The key
branch uses the momentum encoder (m = 0.999), updated after every optimiser
step.  P3/P4/P5 are projected to 256 channels, globally average-pooled and
fused by element-wise mean into one lesion representation; the projection
head is a 2-layer MLP (256 → 128) with `τ = 0.07`, `λ_reg = 1e-4`.  CMCL is
used during training only.

**MedHead** (Methods 3.4, Table 4, Appendix B) — P3/P4/P5 → per-scale 1×1
conv → 256 channels → three parallel attention branches (scale-aware,
spatial-aware deformable convolution with groups = 8, task-aware Dynamic
ReLU), fused by concatenation + 1×1 convolution + SiLU with a weighted
residual (γ = 0.2).  Classification heads are dataset-specific
(BraTS2021: 1, LUNA16: 1, VinDr-CXR: 14 channels); regression predicts the
four-parameter box representation.

The joint objective (Eq. 5/6) is
`L_total = gamma_SR·L_SR + gamma_CMCL·L_CMCL + gamma_Det·L_Det`
with all three coefficients set to 1.0 (Section 4.3).

---

## 4. Datasets and split

BraTS2021 (MRI), LUNA16 (CT) and VinDr-CXR (chest X-ray) are publicly
available and are not redistributed.  `scripts/preprocess.py` implements the
preprocessing of Section 4.2 / Appendix D: 1 mm isotropic resampling for
CT/MRI volumes (linear interpolation for images, nearest-neighbour for
masks), T1ce/T2/FLAIR stacking with enhancing-tumour targets for BraTS2021,
HU clipping with slice-wise nodule cross-sections for LUNA16, percentile
clipping with the native 14-class boxes for VinDr-CXR; 640×640 HR letterbox
and bicubic 160×160 LR (×4 scale).

The data partition is a **fixed patient-level 7:1:2 split** performed before
slice extraction (Section 4.1).  It is generated once by
`scripts/preprocess.py` under an independent fixed split seed
(`--split-seed`, default 42) and persisted as
`splits/{train,val,test}_patients.txt`; all repeated runs — including the
five optimisation seeds 42–46, which vary stochastic optimisation only —
read the same persisted manifest.  The BraTS2021 manifest shipped in
`data/splits/brats2021/` is 876 / 125 / 250 patients (70 / 10 / 20 % of
1,251).

---

## 5. Running the pipeline

```bash
# 1. Preprocessing (Appendix D)
python -m scripts.preprocess --dataset brats2021 \
    --raw /path/to/BraTS2021 --out /path/to/prepared/brats2021 \
    --split-seed 42
python -m scripts.preprocess --dataset luna16 \
    --raw /path/to/LUNA16 --out /path/to/prepared/luna16 --split-seed 42
python -m scripts.preprocess --dataset vindrcxr \
    --raw /path/to/VinDr-CXR --out /path/to/prepared/vindrcxr --split-seed 42

# 2. Joint training (Section 4.3); one run per seed
python -m scripts.train --config configs/default.yaml \
    --data-root /path/to/prepared --seed 42

# 3. Evaluation (Section 4.3)
python -m scripts.evaluate --config configs/default.yaml \
    --data-root /path/to/prepared --dataset brats2021 --split test \
    --checkpoint runs/exp/seed42/best.pt

# 4. Inference
python -m scripts.inference --config configs/default.yaml \
    --checkpoint runs/exp/seed42/best.pt --input /path/to/lr \
    --dataset brats2021 --output detections.json
```

---

## 6. License

Released under the MIT license for review purposes; the BasicSR-derived
components in `modules/_arch_utils.py` and `modules/rrdbnet_arch.py` remain
under the Apache License 2.0 (see `LICENSE_BasicSR.txt`).
