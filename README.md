# MedSRDet — Anonymous Reviewer Repository

Anonymous companion repository for the manuscript **"MedSRDet: A Unified
Framework Jointly Coupling Frequency-Aware Super-Resolution, Lesion-Aware
Representation Regularization, and Small-Lesion Detection for Heterogeneous
Medical Images"** (Biomedical Signal Processing and Control, under review).

This repository provides reference implementations of the three modules that
constitute the methodological contribution of the manuscript:

| Module | Manuscript | Source |
|---|---|---|
| **MFASR** — Medical Frequency-Aware Super-Resolution | Methods 3.2, Table 2 | `modules/mfasrnet_arch.py`, `modules/rrdbnet_arch.py`, `modules/loss_mfasr.py` |
| **CMCL** — Cross-view Modality-robust Contrastive Learning | Methods 3.3, Table 3, Appendix B/C | `modules/cmcl.py` |
| **MedHead** — Medical small-lesion detection head | Methods 3.4, Table 4, Appendix B | `modules/medhead.py` |

The implementations follow the manuscript descriptions; deviations are not
intentional and any discrepancy should be reported through the review process.

---

## 1. Environment

Reference environment (manuscript Table 5): Ubuntu 20.04 LTS, Python 3.8.12,
PyTorch 2.2.0, CUDA 11.3 + cuDNN 8.2, trained on NVIDIA RTX 4090 (24 GB) and
8×A100-SXM4-80GB.

```bash
pip install -r requirements.txt
```

The modules depend only on `torch` / `torchvision`; the shared YOLOv12n
feature encoder (Appendix B) is loaded from the installed `ultralytics`
package and is not bundled here.

---

## 2. Repository layout

```
MedSRDet/
├── modules/                 # reference implementations of the three modules
│   ├── __init__.py          # public API
│   ├── mfasrnet_arch.py     # MFASRNet: RRDB trunk, x4, two-stage NN upsampling
│   ├── rrdbnet_arch.py      # RRDB / ResidualDenseBlock reconstruction backbone
│   ├── loss_mfasr.py        # L_SR = L_pix + 0.1 L_freq + 0.05 L_perc
│   ├── cmcl.py              # CMCL contrastive regularizer (training-time only)
│   ├── medhead.py           # MedHead parallel multi-attention detection head
│   └── _arch_utils.py       # BasicSR-compatible helpers (Apache-2.0)
├── scripts/
│   ├── preprocess.py        # Appendix D: preprocessing + 2D target construction
│   ├── train.py             # Section 4.3 / Appendix E: joint optimisation
│   ├── inference.py         # deployment pathway (CMCL removed)
│   ├── evaluate.py          # mAP@0.5, precision, recall
│   └── selfcheck.py         # module audit against Tables 2/3/4 / Appendix B
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

### CMCL (Methods 3.3, Table 3, Appendix B/C)

* **Positive pair**: the augmented view and the MFASR-enhanced view **of the
  same annotated lesion instance**. CMCL does **not** build cross-modal
  positives (CT↔MRI, CT↔X-ray, MRI↔X-ray) and does not require a unified
  disease taxonomy.
* **Negative samples**: other lesion instances in the mini-batch plus the
  momentum memory queue (`K = 65,536`).
* **Feature aggregation** (Appendix B, Table B.1): P3/P4/P5 are independently
  projected to 256 channels by dedicated 1×1 convolutions, spatially
  aggregated by global average pooling, and the three 256-d vectors are fused
  by **element-wise mean** into one lesion representation.
* **Projection head**: 2-layer MLP, ReLU, dropout 0.2, `256 → 128`, followed
  by ℓ2 normalization.
* **Momentum encoder** (Table 3): the projection head is mirrored as `g_φm`
  by EMA; the shared feature encoder can additionally be mirrored as `E_θm`
  via `CMCL.attach_momentum_backbone()`. `E_θm` encodes the key view only and
  never replaces the online backbone, so the parameter sharing of Methods 3.1
  is preserved. Momentum `m = 0.999`.
* **Hyper-parameters**: `τ = 0.07`, `λ_reg = 1e-4` (Frobenius norm on the
  projection-head weights).
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
projected input by a weighted residual (`γ = 0.2`). Detection heads are
dataset-specific 1×1 convolutions for classification (`N_cls^(d)`) and a shared
1×1 convolution for the 4-parameter regression `(dx, dy, dw, dh)`.

---

## 4. Datasets and splits

Experiments use three publicly available datasets — **BraTS2021** (MRI),
**LUNA16** (CT) and **VinDr-CXR** (chest X-ray). They are publicly available
and are therefore not redistributed here; preprocessing and 2D
detection-target construction follow Section 4.2 and Appendix D of the
manuscript and are implemented by `scripts/preprocess.py` (T1ce / T2 / FLAIR
stacking with ET targets for BraTS2021, HU clipping and slice-wise nodule
cross-sections for LUNA16, percentile clipping and three-channel replication
for VinDr-CXR; 640×640 HR letterbox, bicubic 160×160 LR, ×4 scale).

The **fixed patient-level 7:1:2 partition** is provided in
`data/splits/brats2021/`:

| subset | patients | share |
|---|---|---|
| train | 876 | 70.0 % |
| validation | 125 | 10.0 % |
| test | 250 | 20.0 % |
| **total** | **1,251** | 100 % |

As stated in Section 4.1, the same partition is retained across all five runs
(seeds 42–46, see `configs/default.yaml`) and for all compared methods; only
the optimisation seed changes. Validation/test subsets keep every eligible
slice (no positive–negative balancing).

---

## 5. Reproducing the pipeline

Run from the repository root:

```bash
# 1. Preprocessing and 2D detection-target construction (Appendix D)
python -m scripts.preprocess --dataset brats2021 \
    --raw /path/to/BraTS2021 --out /path/to/prepared/brats2021
python -m scripts.preprocess --dataset luna16 \
    --raw /path/to/LUNA16 --out /path/to/prepared/luna16
python -m scripts.preprocess --dataset vindrcxr \
    --raw /path/to/VinDr-CXR --out /path/to/prepared/vindrcxr

# 2. Joint training (Section 4.3); one run per seed
python -m scripts.train --config configs/default.yaml \
    --data-root /path/to/prepared --seed 42

# 3. Evaluation after checkpoint selection (Section 4.3)
python -m scripts.evaluate --config configs/default.yaml \
    --data-root /path/to/prepared --dataset brats2021 --split test \
    --checkpoint runs/exp/seed42/best.pt

# 4. Inference on a directory of low-resolution images
python -m scripts.inference --config configs/default.yaml \
    --checkpoint runs/exp/seed42/best.pt --input /path/to/lr --output detections.json
```

`scripts/train.py` builds **mixed-modal mini-batches** that jointly sample
BraTS2021, LUNA16 and VinDr-CXR (Section 3.3), optimises
`L_total = L_SR + L_CMCL + L_Det` with γ = 1.0 each (Eq. 6) using AdamW
(lr 1e-4, weight decay 1e-2), cosine annealing, 300 epochs, batch size 16 and
AMP (Table 5), applies the 1:1 positive:negative slice re-sampling of
Appendix D.4, computes the BCE + CIoU detection objective of Appendix E, and
keeps the checkpoint with the highest validation mAP@0.5. All CMCL-specific
components are excluded from the inference pathway (Section 3.5).

A dataset-free configuration audit is available via `scripts/selfcheck.py`.

---

## 6. License

Released under the MIT license for review purposes; the BasicSR-derived
components in `modules/_arch_utils.py` and `modules/rrdbnet_arch.py` remain
under the Apache License 2.0 (see `LICENSE_BasicSR.txt`).
