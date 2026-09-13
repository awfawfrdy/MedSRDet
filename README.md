# MedSRDet — Anonymous Reviewer Repository

Anonymous companion repository for the manuscript **"MedSRDet: A Unified
Framework of Frequency-Aware Super-Resolution and Modality-Robust
Representation Regularization for Small Lesion Detection"** (Biomedical
Signal Processing and Control, under review).

This repository provides a reference implementation of the MedSRDet training
and inference pipeline:

| Component | Manuscript | Source |
|---|---|---|
| **MFASR** — Medical Frequency-Aware Super-Resolution | Methods 3.2, Table 2 | `modules/mfasrnet_arch.py`, `modules/rrdbnet_arch.py`, `modules/loss_mfasr.py` |
| **CMCL** — Cross-view Modality-robust Contrastive Learning | Methods 3.3, Table 3, Appendix B/C | `modules/cmcl.py`, `scripts/lesion_views.py` |
| **MedHead** — Medical small-lesion detection head | Methods 3.4, Table 4, Appendix B | `modules/medhead.py` |
| **Shared encoder** — YOLOv12n P3/P4/P5 feature extraction | Appendix B | `modules/yolo_encoder.py` |
| **Decoding / mAP** — single bbox decoder, class-aware mAP@0.5 | Appendix E | `modules/detection_utils.py` |
| **Joint training** — L_SR + L_CMCL + L_Det, mixed-modal batches | Section 4.3, Appendix E | `scripts/train.py` |

The implementations follow the manuscript descriptions.  Decisions on details
the manuscript leaves open are marked as *implementation detail* in the code
docstrings and are summarised at the end of this README.

---

## 1. Environment

Reference environment (manuscript Table 5): Ubuntu 20.04 LTS, Python 3.8.12,
PyTorch 2.2.0, CUDA 11.3 + cuDNN 8.2, trained on NVIDIA RTX 4090 (24 GB) and
8×A100-SXM4-80GB.

```bash
pip install -r requirements.txt
```

The pipeline additionally loads the shared YOLOv12n feature encoder from the
installed `ultralytics` package (checkpoint `yolo12n.pt`, resolved
automatically on first use).

---

## 2. Repository layout

```
MedSRDet/
├── modules/                 # reference implementations
│   ├── __init__.py          # public API
│   ├── mfasrnet_arch.py     # MFASRNet: RRDB trunk, x4, two-stage NN upsampling
│   ├── rrdbnet_arch.py      # RRDB / ResidualDenseBlock reconstruction backbone
│   ├── loss_mfasr.py        # L_SR = L_pix + 0.1 L_freq + 0.05 L_perc
│   ├── cmcl.py              # CMCL contrastive regularizer (training-time only)
│   ├── medhead.py           # MedHead parallel multi-attention detection head
│   ├── yolo_encoder.py      # shared YOLOv12n P3/P4/P5 encoder (graph-accurate)
│   ├── detection_utils.py   # shared bbox decoder, NMS, class-aware mAP@0.5
│   └── _arch_utils.py       # BasicSR-compatible helpers (Apache-2.0)
├── scripts/
│   ├── preprocess.py        # Appendix D: 1 mm isotropic resampling, targets
│   ├── lesion_views.py      # Appendix C: CMCL lesion-view construction
│   ├── train.py             # Section 4.3 / Appendix E: joint optimisation
│   ├── inference.py         # deployment pathway (CMCL removed)
│   ├── evaluate.py          # dataset-level class-aware mAP@0.5
│   ├── selfcheck.py         # module audit against Tables 2/3/4 / Appendix B
│   └── smoke_test.py        # end-to-end pipeline checks (synthetic data)
├── configs/
│   └── default.yaml         # experiment configuration; each value cites its
│                            # manuscript section / table
├── data/splits/brats2021/   # fixed patient-level 7:1:2 split manifest
├── requirements.txt
├── LICENSE
└── LICENSE_BasicSR.txt      # Apache-2.0, covers the BasicSR-derived code
```

---

## 3. Module summary

### MFASR (Methods 3.2, Table 2)

RRDB-based reconstruction backbone: `3 → 64` shallow convolution, 23 RRDB
blocks (3 RDBs each, growth channels `gc = 32`, local residual scaling
`β = 0.2`), long skip connection, two-stage nearest-neighbour upsampling
(×2 + 3×3 conv, twice → overall ×4) and a `64 → 3` reconstruction layer.

Reconstruction objective (Eq. 1):

```
L_SR = L_pix + 0.1 * L_freq + 0.05 * L_perc
```

* `L_pix` — L1 between `I_SR` and `I_HR`
* `L_freq` — L1 on the **FFT magnitude spectrum** (magnitude only; no phase
  term, no adversarial objective — see Methods 3.2 and Table 15)
* `L_perc` — L1 on pretrained VGG-19 **ReLU5_4** features (frozen, eval mode)

### CMCL (Methods 3.3, Table 3, Appendix B/C, Appendix C)

* **Positive pair** (Appendix C): for every annotated lesion instance, two
  views built from the same lesion-centred support — an *augmented view*
  (Table C.1 protocol: rotation/translation/scaling/intensity/noise with
  manuscript-fixed probabilities) and an *MFASR-enhanced view* (HR patch →
  bicubic 56×56 → current MFASR ×4 → 224×224).  CMCL never builds cross-modal
  positives (CT↔MRI, CT↔X-ray, MRI↔X-ray) and does not require a unified
  disease taxonomy.
* **Momentum pathway** (Table 3): `q = g_φ(E_θ(x_aug))`,
  `k = g_φm(E_θm(x_sr))`; the key branch is gradient-free; `E_θm`/`g_φm` are
  EMA-updated (`m = 0.999`) after every optimiser step; keys are enqueued into
  the FIFO memory queue (`K = 65,536`).
* **Negative samples**: other lesion instances in the mini-batch plus the
  memory queue.
* **Feature aggregation** (Appendix B, Table B.1): P3/P4/P5 are independently
  projected to 256 channels by dedicated 1×1 convolutions, spatially
  aggregated by global average pooling, and the three 256-d vectors are fused
  by **element-wise mean** into one lesion representation.
* **Projection head**: 2-layer MLP, ReLU, dropout 0.2, `256 → 128`, followed
  by ℓ2 normalization; `τ = 0.07`, `λ_reg = 1e-4`.
* **Training only**: the projection head, momentum encoder and memory queue
  are removed at inference; `forward()` returns a zero scalar in `eval()` mode.

### MedHead (Methods 3.4, Table 4, Appendix B)

P3/P4/P5 → per-scale 1×1 conv → 256 channels → three **parallel** attention
branches:

1. **Scale-aware**: GAP → MLP 256→64→256 → Sigmoid modulation
2. **Spatial-aware**: 3×3 deformable convolution, `groups = 8`, with offset prediction
3. **Task-aware**: Dynamic ReLU with dynamic `α1, α2, β1, β2`

Branch outputs are refined by lightweight 3×3 convolutions, concatenated along
channels, fused by a 1×1 convolution (→256) + SiLU, and combined with the
projected input by a weighted residual (`γ = 0.2`).  Detection heads are
dataset-specific 1×1 convolutions for classification (BraTS2021: 1 channel,
LUNA16: 1 channel, VinDr-CXR: 14 channels — each dataset keeps its own label
space) and a shared 1×1 convolution for the 4-parameter regression
`(dx, dy, dw, dh)`.

### Joint objective (Eq. 5/6, Appendix E)

```
L_total = gamma_SR * L_SR + gamma_CMCL * L_CMCL + gamma_Det * L_Det
L_Det   = lambda_cls * L_cls + lambda_box * L_box     (lambda_* = 1.0)
```

with `gamma_SR = gamma_CMCL = gamma_Det = 1.0` in the primary configuration.
The γ coefficients are applied outside the detection loss; classification
uses BCE in each dataset's own label space (native VinDr-CXR class ids are
preserved end-to-end), regression uses CIoU decoded with the same shared
decoder as evaluation/inference, and per-dataset losses are aggregated with a
sample-weighted mean (Eq. 4).

---

## 4. Datasets and splits

Experiments use three publicly available datasets — **BraTS2021** (MRI),
**LUNA16** (CT) and **VinDr-CXR** (chest X-ray).  They are publicly available
and are therefore not redistributed here.  Preprocessing follows Section 4.2
and Appendix D and is implemented by `scripts/preprocess.py`: 1 mm isotropic
resampling for CT/MRI volumes (image: linear interpolation, mask:
nearest-neighbour; LUNA16 nodule physical coordinates are converted in the
resampled geometry), T1ce/T2/FLAIR stacking with ET targets for BraTS2021, HU
clipping and slice-wise nodule cross-sections for LUNA16, percentile clipping
and three-channel replication with native 14-class boxes for VinDr-CXR;
640×640 HR letterbox, bicubic 160×160 LR, ×4 scale.

### Fixed patient-level 7:1:2 split (Section 4.1)

Following the manuscript, the partition is **fixed at the patient level** and
is **retained across all repeated runs**: the optimisation seeds 42–46 vary
stochastic optimisation only and **never change the data split**.

* The partition is controlled by an **independent fixed split seed**
  (`--split-seed`, default `42`, decoupled from the optimisation seeds).
* The split is generated once by `scripts/preprocess.py` and **persisted** as
  `splits/{train,val,test}_patients.txt` inside each prepared dataset; every
  later preprocessing or training run **reads that same persisted manifest**
  (idempotent — re-running preprocessing never re-generates it).
* An externally provided frozen manifest
  (`--split-manifest-root data/splits`) takes precedence; the split source is
  recorded in each dataset's `protocol.json`.
* `scripts/train.py` reads the manifests only and verifies before training
  that train/val/test are complete and patient-disjoint
  (`verify_fixed_splits`).
* `scripts/selfcheck.py` and `scripts/smoke_test.py` include checks that all
  five training seeds (42–46) read an **identical patient split**.

The fixed patient-level partition currently shipped in this repository is
`data/splits/brats2021/`:

| subset | patients | share |
|---|---|---|
| train | 876 | 70.0 % |
| validation | 125 | 10.0 % |
| test | 250 | 20.0 % |
| **total** | **1,251** | 100 % |

**Scope note.**  These split lists are produced deterministically by the
released implementation so that every repeated run uses an identical
patient-level 7:1:2 partition, exactly as described in the manuscript.  For
LUNA16 and VinDr-CXR the manifests are generated by this implementation at
preprocessing time (from the independent split seed and the patient IDs found
in the raw data) and persisted; they are implementation-level artifacts for
reproducibility and are **not claimed to be the historical manifests** used to
produce the reported numbers.

Negative-slice sampling (Appendix D.4): the 1:1 positive:negative ratio with
per-epoch re-randomisation applies **only to BraTS2021/LUNA16 training
slices**; validation and test keep every eligible slice; VinDr-CXR keeps all
image-level annotations.  This protocol is encoded in `SliceDataset`
(`scripts/train.py`) and verified by `scripts/smoke_test.py`.

---

## 5. Reproducing the pipeline

Run from the repository root:

```bash
# 1. Preprocessing and 2D detection-target construction (Appendix D).
#    --split-seed is the INDEPENDENT fixed split seed (default 42); the
#    partition is persisted and reused by every later run.  The optimisation
#    seeds 42-46 of step 2 never change the split.
python -m scripts.preprocess --dataset brats2021 \
    --raw /path/to/BraTS2021 --out /path/to/prepared/brats2021 \
    --split-manifest-root data/splits --split-seed 42
python -m scripts.preprocess --dataset luna16 \
    --raw /path/to/LUNA16 --out /path/to/prepared/luna16 --split-seed 42
python -m scripts.preprocess --dataset vindrcxr \
    --raw /path/to/VinDr-CXR --out /path/to/prepared/vindrcxr --split-seed 42

# 2. Joint training (Section 4.3); one run per seed
python -m scripts.train --config configs/default.yaml \
    --data-root /path/to/prepared --seed 42

# 3. Evaluation after checkpoint selection (Section 4.3)
python -m scripts.evaluate --config configs/default.yaml \
    --data-root /path/to/prepared --dataset brats2021 --split test \
    --checkpoint runs/exp/seed42/best.pt

# 4. Inference on a directory of low-resolution images
python -m scripts.inference --config configs/default.yaml \
    --checkpoint runs/exp/seed42/best.pt --input /path/to/lr \
    --dataset brats2021 --output detections.json
```

`scripts/train.py` builds **mixed-modal mini-batches** of exactly 16 samples
covering BraTS2021, LUNA16 and VinDr-CXR (Section 3.3; the extra sample when
16 is not divisible by three rotates across datasets), constructs the
Appendix-C CMCL views for every annotated lesion instance, optimises
`L_total = L_SR + L_CMCL + L_Det` with γ = 1.0 each (Eq. 6) using AdamW
(lr 1e-4, weight decay 1e-2), cosine annealing, 300 epochs and AMP (Table 5),
performs the EMA momentum-encoder update after every optimiser step, computes
the BCE + CIoU detection objective of Appendix E with dataset-specific heads,
and keeps the checkpoint with the highest validation mAP@0.5.  All
CMCL-specific components are excluded from the inference pathway (Section
3.5).

`scripts/inference.py` writes real detection results:

```json
{
  "example.png": [
    {"class_id": 0, "score": 0.87, "bbox_xyxy": [212.5, 198.0, 274.5, 260.5]}
  ]
}
```

Two dataset-free audits are available:

```bash
python -m scripts.selfcheck     # module audit against Tables 2/3/4 / App. B
python -m scripts.smoke_test    # end-to-end pipeline checks (synthetic data)
```

---

## 6. Implementation details not fixed by the manuscript

The following details are not pinned down by the manuscript and are
implementation decisions, documented here and in the code docstrings:

* **Box parameterisation / decoding** (`modules/detection_utils.py`): the
  anchor-free decoding `(sigmoid(dx)*2-0.5+gx)*stride`,
  `(sigmoid(dw)*2)²*stride` used identically by training, evaluation and
  inference.  The manuscript fixes the four-parameter regression target but
  not the decoding rule.
* **Target assignment**: each ground-truth object is assigned to the grid cell
  containing its centre (single positive per object per level).
* **mAP@0.5**: dataset-level, class-aware AP with COCO 101-point
  interpolation; a class contributes only if it has ground-truth instances in
  the split.
* **CMCL view batching**: all lesion instances of a mini-batch are encoded in
  one contrastive step; the per-instance augmentation RNG is derived from
  (seed, epoch, instance index) for reproducibility.
* **Dynamic-ReLU coefficient network hidden width** = 64; MedHead attention
  branches are shared across pyramid levels (FPN-style head weight sharing).
* **Mixed-modal quota rotation**: with batch 16 and three datasets the quotas
  are 5+5+6 with the extra sample rotating per batch.

---

## 7. License

Released under the MIT license for review purposes; the BasicSR-derived
components in `modules/_arch_utils.py` and `modules/rrdbnet_arch.py` remain
under the Apache License 2.0 (see `LICENSE_BasicSR.txt`).
