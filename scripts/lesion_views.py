"""CMCL lesion-view construction (Section 3.3, Appendix C / Table C.1).

For every annotated lesion instance two views are built from the SAME
lesion-centred spatial support:

* **augmented view**  — photometric / geometric augmentation of the HR patch;
* **MFASR-enhanced view** — the HR patch bicubically downsampled to 56 x 56
  and reconstructed by the *current* MFASR network (x4 -> 224 x 224).

Both views are positive partners of the same lesion instance; CMCL never
pairs lesions from different images / modalities.

Crop rule (Table C.1)
---------------------
* square side ``s = 1.5 * max(w, h)`` of the GT box (i.e. a 25 % margin on
  both sides of the longer box side);
* centred on the box centre;
* out-of-bounds regions use border replication;
* the crop is resized to 224 x 224 with bilinear interpolation.

Augmented view (Table C.1) — all probabilities / ranges are manuscript-fixed:
    rotation        p=0.5, [-10, +10] deg
    translation     p=0.5, max 5 % (of the patch size, both axes)
    isotropic scale p=0.5, [0.90, 1.10]
    intensity scale p=0.5, [0.90, 1.10]
    intensity shift p=0.5, [-0.05, +0.05]
    Gaussian noise  p=0.3, sigma ~ U(0, 0.02)
    clip to [0, 1] after augmentation

The per-instance RNG is derived from (seed, epoch, image index, instance
index) so the view construction is reproducible.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch

# Manuscript constants (Table C.1)
PATCH_SIZE_DEFAULT = 224
DOWNSAMPLE_SIZE_DEFAULT = 56       # 224 / scale(4)
MARGIN_RATIO_DEFAULT = 0.25        # crop side = 1.5 * max(w, h)


def crop_lesion_patch(image: np.ndarray, box_cxcywh_norm, img_size: int = 640,
                      margin_ratio: float = MARGIN_RATIO_DEFAULT,
                      patch_size: int = PATCH_SIZE_DEFAULT) -> np.ndarray:
    """Lesion-centred square crop with border replication (Table C.1).

    Args:
        image: [H, W, C] float32 in [0, 1] (the HR image).
        box_cxcywh_norm: (cx, cy, w, h) normalised YOLO box.
        img_size: HR image size used to normalise the box.
        margin_ratio: extra margin on each side of the longer box side.
        patch_size: output patch resolution (224).

    Returns:
        [patch_size, patch_size, C] float32 in [0, 1].
    """
    cx, cy, bw, bh = (float(v) for v in box_cxcywh_norm)
    cx_px, cy_px = cx * img_size, cy * img_size
    w_px, h_px = bw * img_size, bh * img_size

    side = (1.0 + 2.0 * margin_ratio) * max(w_px, h_px)
    side = max(side, 4.0)                       # degenerate-box guard

    x0 = cx_px - side / 2.0
    y0 = cy_px - side / 2.0

    H, W = image.shape[:2]
    # Map the (possibly out-of-bounds) float window onto integer indices and
    # sample with border replication: source coordinates are clamped to the
    # image, then the crop is taken and resized.
    ix0 = int(math.floor(x0))
    iy0 = int(math.floor(y0))
    side_i = int(math.ceil(side))

    # Nearest valid source window (clamped into the image).
    sx0 = min(max(ix0, 0), max(W - 1, 0))
    sy0 = min(max(iy0, 0), max(H - 1, 0))
    sx1 = min(max(ix0 + side_i, sx0 + 1), W)
    sy1 = min(max(iy0 + side_i, sy0 + 1), H)

    crop = image[sy0:sy1, sx0:sx1]

    # Pad the crop so that it covers the full (unclamped) window size, using
    # border replication on the appropriate edges.
    pad_left = sx0 - ix0
    pad_top = sy0 - iy0
    pad_right = (ix0 + side_i) - sx1
    pad_bottom = (iy0 + side_i) - sy1
    if any(p > 0 for p in (pad_left, pad_top, pad_right, pad_bottom)):
        crop = cv2.copyMakeBorder(
            crop, max(pad_top, 0), max(pad_bottom, 0),
            max(pad_left, 0), max(pad_right, 0),
            borderType=cv2.BORDER_REPLICATE,
        )

    crop = cv2.resize(crop, (patch_size, patch_size),
                      interpolation=cv2.INTER_LINEAR)   # bilinear (Table C.1)
    return np.ascontiguousarray(crop, dtype=np.float32)


def augment_lesion_patch(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply the Table C.1 augmentation protocol to one 224 x 224 patch."""
    out = patch.copy()
    size = out.shape[0]

    # random rotation: p=0.5, [-10, +10] deg (includes translation here via
    # the affine matrix, then translation is applied on top with its own p).
    def _affine(angle=0.0, scale=1.0, tx=0.0, ty=0.0):
        mat = cv2.getRotationMatrix2D((size / 2.0, size / 2.0), angle, scale)
        mat[0, 2] += tx * size
        mat[1, 2] += ty * size
        return cv2.warpAffine(out, mat, (size, size),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    if rng.random() < 0.5:
        out = _affine(angle=float(rng.uniform(-10.0, 10.0)))
    if rng.random() < 0.5:                          # translation, max 5 %
        out = _affine(tx=float(rng.uniform(-0.05, 0.05)),
                      ty=float(rng.uniform(-0.05, 0.05)))
    if rng.random() < 0.5:                          # isotropic scaling
        out = _affine(scale=float(rng.uniform(0.90, 1.10)))

    if rng.random() < 0.5:                          # intensity scaling
        out = out * float(rng.uniform(0.90, 1.10))
    if rng.random() < 0.5:                          # intensity shift
        out = out + float(rng.uniform(-0.05, 0.05))
    if rng.random() < 0.3:                          # Gaussian noise
        sigma = float(rng.uniform(0.0, 0.02))
        out = out + rng.normal(0.0, sigma, size=out.shape).astype(np.float32)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def build_cmcl_views(hr_batch: torch.Tensor, boxes: torch.Tensor,
                     classes: torch.Tensor, batch_idx: torch.Tensor,
                     seed: int = 42, epoch: int = 0,
                     patch_size: int = PATCH_SIZE_DEFAULT,
                     down_size: int = DOWNSAMPLE_SIZE_DEFAULT):
    """Construct the CMCL positive-pair views for every lesion instance.

    Args:
        hr_batch: [B, 3, H, W] HR images in [0, 1].
        boxes: [N, 4] normalised (cx, cy, w, h) of all GT boxes in the batch.
        classes: [N] class ids (kept for interface symmetry; CMCL is
            class-agnostic and pairs by instance index only).
        batch_idx: [N] image index of each GT box.
        seed / epoch: reproducibility of the augmentation draws.

    Returns:
        (augmented_views [N, 3, 224, 224], low_res_inputs [N, 3, 56, 56]).
        The MFASR-enhanced view itself is produced by the training loop via
        ``mfasar(low_res_inputs)`` (x4 -> 224 x 224).
    """
    if boxes.numel() == 0:
        dev = hr_batch.device
        z = torch.zeros((0, 3, patch_size, patch_size), device=dev)
        return z, torch.zeros((0, 3, down_size, down_size), device=dev)

    hr_np = hr_batch.detach().cpu().numpy()
    img_size = hr_np.shape[-1]

    aug_list, low_list = [], []
    for i in range(boxes.shape[0]):
        b = int(batch_idx[i].item())
        img = hr_np[b].transpose(1, 2, 0)           # [H, W, C]
        patch = crop_lesion_patch(img, boxes[i].tolist(), img_size=img_size,
                                  patch_size=patch_size)
        rng = np.random.default_rng(seed * 1_000_003 + epoch * 10_007 + i)
        aug = augment_lesion_patch(patch, rng)

        low = cv2.resize(patch, (down_size, down_size),
                         interpolation=cv2.INTER_CUBIC)   # bicubic (Table C.1)
        aug_list.append(aug.transpose(2, 0, 1))
        low_list.append(low.astype(np.float32).transpose(2, 0, 1))

    aug_t = torch.from_numpy(np.ascontiguousarray(np.stack(aug_list)))
    low_t = torch.from_numpy(np.ascontiguousarray(np.stack(low_list)))
    return aug_t.to(hr_batch.device), low_t.to(hr_batch.device)
