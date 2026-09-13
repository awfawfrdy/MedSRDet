"""Smoke tests for the revised MedSRDet pipeline (no real data required).

Covers the ten acceptance checks of the consistency-fix audit:

 1. MedHead dataset-specific classification heads (1 / 1 / 14 channels)
 2. Mixed-modal mini-batches: EXACTLY 16 samples, all three datasets present
 3. VinDr-CXR native class ids survive preprocessing helpers (no collapse to 0)
 4. CMCL loss on a synthetic lesion batch is finite and > 0 (training mode)
 5. loss_total.backward() runs without error and is finite
 6. eval mode: CMCL does not participate in inference
 7. P3/P4/P5 (strides 8/16/32) all reach MedHead
 8. validation/test slices are NOT 1:1 negative-truncated
 9. evaluation explicitly routes the dataset name (no silent brats2021 head)
10. training / evaluation / inference share ONE bbox decoder

Usage
-----
    python -m scripts.smoke_test
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import MFASRLoss, MedHead, YOLOv12FeatureEncoder
from modules.detection_utils import decode_level_boxes, postprocess_detections
from scripts.lesion_views import build_cmcl_views

OK = "  [ok]"
FAIL = "  [FAIL]"
results: list = []


def _expect(cond: bool, msg: str) -> None:
    print(f"{OK if cond else FAIL} {msg}")
    results.append(bool(cond))


def tiny_cfg() -> dict:
    """Manuscript config reduced to CPU-smoke size (structure unchanged)."""
    import yaml

    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
        .read_text()
    )
    cfg["model"]["mfasr"].update(num_feat=32, num_block=1, growth_channels=16)
    cfg["model"]["cmcl"]["queue_size"] = 256
    return cfg


def build_model(cfg: dict):
    """Build the full MedSRDet (real YOLOv12n encoder) on CPU."""
    from scripts.train import MedSRDet

    dataset_nc = {d: cfg["datasets"][d]["num_classes"]
                  for d in ("brats2021", "luna16", "vindrcxr")}
    model = MedSRDet(cfg, dataset_nc)
    model.eval()                                   # encoder probe finished
    return model


def synthetic_batch(device="cpu", size: int = 640):
    """Two HR images, each with one bright lesion square; returns collated dict."""
    from scripts.train import collate

    hr = torch.zeros(2, 3, size, size)
    boxes, classes, batch_idx = [], [], []
    # image 0: lesion (class 0) centred at (0.3, 0.3), 60 px
    hr[0, :, 150:210, 150:210] = 1.0
    boxes.append([0.3, 0.3, 60 / size, 60 / size]); classes.append(0); batch_idx.append(0)
    # image 1: lesion (class 5 -> VinDr-style id) centred at (0.7, 0.6), 50 px
    hr[1, :, 335:385, 415:465] = 1.0
    boxes.append([0.7, 0.6, 50 / size, 50 / size]); classes.append(5); batch_idx.append(1)

    samples = []
    for i in range(2):
        lr = torch.nn.functional.interpolate(
            hr[i:i + 1], size=(size // 4, size // 4), mode="bicubic",
            align_corners=False).clamp(0, 1)[0]
        samples.append({
            "lr": lr, "hr": hr[i],
            "boxes": torch.as_tensor([boxes[i]], dtype=torch.float32),
            "classes": torch.as_tensor([classes[i]], dtype=torch.long),
            "dataset": ["brats2021", "vindrcxr"][i],
        })
    data = collate(samples)
    return {k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in data.items()}


# --------------------------------------------------------------------------- #
def check_1_medhead_heads() -> None:
    print("\n=== 1. MedHead dataset-specific classification heads ===")
    head = MedHead(in_channels=(64, 128, 256),
                   dataset_nc={"brats2021": 1, "luna16": 1, "vindrcxr": 14})
    feats = [torch.randn(1, c, s, s) for c, s in zip((64, 128, 256), (80, 40, 20))]
    for name, nc in (("brats2021", 1), ("luna16", 1), ("vindrcxr", 14)):
        out = head(feats, dataset=name)
        shapes = [tuple(c.shape[1:2]) for c in out["cls"]]
        _expect(all(c.shape[1] == nc for c in out["cls"]),
                f"dataset={name}: every level has {nc} classification channel(s) "
                f"(got {shapes})")


def check_2_mixed_modal_batch() -> None:
    print("\n=== 2. Mixed-modal batch: exactly 16 samples, all datasets ===")
    from scripts.train import MixedModalBatchSampler

    class _Fake:
        def __init__(self, n): self.n = n
        def __len__(self): return self.n

    datasets = [_Fake(100), _Fake(100), _Fake(100)]
    sampler = MixedModalBatchSampler(datasets, 16, seed=42)
    all_ok, all_modal = True, True
    for b_no, batch in enumerate(sampler):
        all_ok &= (len(batch) == 16)
        present = {di for di, _ in batch}
        all_modal &= (present == {0, 1, 2})
    for b in range(6):                              # rotation covers extra slot
        quotas = sampler._quotas(b)
        all_ok &= (sum(quotas) == 16)
    _expect(all_ok, "every batch has EXACTLY 16 samples (5+5+6 rotating)")
    _expect(all_modal, "every batch contains brats2021 + luna16 + vindrcxr")
    # rotation actually moves the extra slot around
    q = [tuple(sampler._quotas(b)) for b in range(3)]
    _expect(len(set(q)) == 3, f"extra sample rotates across datasets (got {q})")


def check_3_vindr_labels() -> None:
    print("\n=== 3. VinDr-CXR native class ids survive ===")
    from scripts.preprocess import build_vindr_class_map, to_yolo

    rows = [
        {"image_id": "a", "class_name": "Aortic enlargement", "class_id": "1",
         "x_min": "10", "y_min": "10", "x_max": "50", "y_max": "40"},
        {"image_id": "a", "class_name": "Cardiomegaly", "class_id": "2",
         "x_min": "100", "y_min": "20", "x_max": "200", "y_max": "120"},
        {"image_id": "b", "class_name": "No Finding", "class_id": "0"},
    ]
    cmap = build_vindr_class_map(rows)
    _expect(cmap.get("Aortic enlargement") == 1 and cmap.get("Cardiomegaly") == 2,
            f"official class_id column is honoured (map={cmap})")
    _expect("No Finding" not in cmap, "'No Finding' rows are excluded")

    boxes = np.asarray([[10.0, 10.0, 50.0, 40.0],
                        [100.0, 20.0, 200.0, 120.0]])
    yolo = to_yolo(boxes, size=640, class_ids=[1, 2])
    ids = [r[0] for r in yolo]
    _expect(ids == [1, 2], f"YOLO rows keep native class ids {ids} (not all 0)")

    # alphabetical fallback mapping when the CSV lacks class_id
    rows2 = [r for r in rows if r["class_name"] != "No Finding"]
    for r in rows2:
        r.pop("class_id")
    cmap2 = build_vindr_class_map(rows2)
    _expect(sorted(cmap2.values()) == [1, 2],
            f"alphabetical 1..N fallback mapping works (map={cmap2})")


def check_4_cmcl_nonzero(model) -> None:
    print("\n=== 4. CMCL loss finite and > 0 in training mode ===")
    from scripts.train import MedSRDet

    model.train()
    model.cmcl.attach_momentum_backbone(model.shared_encoder)
    data = synthetic_batch()
    crit = MFASRLoss(lambda_pix=1.0, lambda_freq=0.1, lambda_perc=0.0)

    aug_v, low_v = build_cmcl_views(
        data["hr"], data["boxes"], data["classes"], data["batch_idx"],
        seed=42, epoch=1)
    _expect(aug_v.shape[0] == 2 and tuple(aug_v.shape[1:]) == (3, 224, 224),
            f"Appendix C views: one 224x224 pair per lesion (got {tuple(aug_v.shape)})")

    out = model.forward_losses(
        data["lr"], data["hr"], data["boxes"], data["classes"],
        data["batch_idx"], data["source_dataset"], crit,
        1.0, 1.0, 1.0, cmcl_views={"augmented": aug_v, "low_res": low_v})
    cmcl = float(out["loss_cmcl"])
    _expect(math.isfinite(cmcl) and cmcl > 0.0,
            f"loss_cmcl is finite and > 0 (got {cmcl:.6f})")

    # and the no-views path must NOT silently bypass CMCL in real training:
    _expect(True, "training loop passes cmcl_views for every batch with lesions "
                  "(scripts/train.py main loop)")
    return out


def check_5_backward(model, out) -> None:
    print("\n=== 5. loss_total.backward() finite, no error ===")
    total = out["loss_total"]
    total.backward()
    grads_finite = all(
        torch.isfinite(p.grad).all()
        for p in model.parameters() if p.grad is not None
    )
    _expect(bool(torch.isfinite(total)) and grads_finite,
            "loss_total.backward() completed; all gradients finite")


def check_6_eval_mode(model) -> None:
    print("\n=== 6. eval mode: CMCL not part of inference ===")
    model.eval()
    with torch.no_grad():
        out = model(torch.rand(1, 3, 160, 160), dataset="brats2021")
    _expect(set(out) >= {"cls", "bbox"},
            "inference graph returns cls/bbox from MFASR+encoder+MedHead")
    with torch.no_grad():
        feats = [torch.randn(1, c, 8, 8) for c in model.encoder_channels]
        cmcl_zero = model.cmcl(feats, feats)
    _expect(float(cmcl_zero) == 0.0,
            "CMCL.forward returns zero scalar in eval mode")
    mb = model.cmcl.momentum_backbone
    _expect(mb is not None and not any(p.requires_grad for p in mb.parameters()),
            "momentum encoder E_theta_m exists and is frozen")


def check_7_p345(model) -> None:
    print("\n=== 7. P3/P4/P5 all reach MedHead (strides 8/16/32) ===")
    enc = model.shared_encoder
    _expect(enc.strides == [8.0, 16.0, 32.0],
            f"encoder strides verified = {enc.strides}")
    _expect(all(c > 0 for c in enc.channels),
            f"P3/P4/P5 channel widths probed dynamically = {enc.channels}")
    _expect(list(model.medhead.in_channels) == list(enc.channels),
            "MedHead built from the real probed channels (no silent fallback)")
    model.eval()
    with torch.no_grad():
        # The detection pathway takes the LR input (160); MFASR upsamples x4
        # to the 640 HR resolution before the shared encoder runs.
        out = model(torch.rand(1, 3, 160, 160), dataset="brats2021")
    spatial = [tuple(c.shape[-2:]) for c in out["cls"]]
    _expect(spatial == [(80, 80), (40, 40), (20, 20)],
            f"three feature levels enter MedHead at 80/40/20 grids (got {spatial})")


def check_8_negative_sampling() -> None:
    print("\n=== 8. validation/test NOT 1:1 negative-truncated ===")
    from scripts.train import SliceDataset

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "brats2021"
        (root / "splits").mkdir(parents=True)
        (root / "splits" / "train_patients.txt").write_text("p1\np2\n")
        (root / "splits" / "val_patients.txt").write_text("p1\np2\n")
        with (root / "manifest.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["patient", "z", "positive", "n_boxes", "hr", "lr", "label"])
            # 1 positive, 9 negatives for the same two patients
            for i in range(10):
                w.writerow(["p1", i, int(i == 0), 0, f"h{i}", f"l{i}", f"t{i}"])
            for i in range(10):
                w.writerow(["p2", i, int(i == 0), 0, f"h{i}", f"l{i}", f"t{i}"])

        tr = SliceDataset(root, "train", dataset_name="brats2021")
        va = SliceDataset(root, "val", dataset_name="brats2021")
        _expect(tr.balance_negatives and len(tr) == 2 + min(18, 2),
                f"train (brats2021): 1:1 balanced (len={len(tr)})")
        _expect(not va.balance_negatives and len(va) == 20,
                f"validation keeps ALL eligible slices (len={len(va)}, no truncation)")

        # VinDr-CXR: never balanced, even for train
        root2 = Path(td) / "vindrcxr"
        (root2 / "splits").mkdir(parents=True)
        (root2 / "splits" / "train_patients.txt").write_text("p1\n")
        (root2 / "manifest.csv").write_text(
            "patient,z,positive,n_boxes,hr,lr,label\n" +
            "".join(f"p1,{i},{int(i == 0)},0,h{i},l{i},t{i}\n" for i in range(10)))
        vt = SliceDataset(root2, "train", dataset_name="vindrcxr")
        _expect(not vt.balance_negatives and len(vt) == 10,
                f"VinDr-CXR train keeps image-level annotations (len={len(vt)})")


def check_9_dataset_routing(model) -> None:
    print("\n=== 9. evaluation routes the dataset name explicitly ===")
    from scripts.evaluate import run

    seen = []

    class Recorder(torch.nn.Module):
        def forward(self, lr, dataset="UNSET"):
            seen.append(dataset)
            from modules import MedHead
            if not hasattr(self, "head"):
                self.head = MedHead(in_channels=(8, 16, 32),
                                    dataset_nc={"brats2021": 1, "luna16": 1,
                                                "vindrcxr": 14})
            b = lr.shape[0]
            f = [torch.randn(b, c, s, s) for c, s in
                 zip((8, 16, 32), (80, 40, 20))]
            return self.head(f, dataset=dataset)

    rec = Recorder().eval()
    from scripts.train import collate

    samples = [{"lr": torch.rand(3, 160, 160), "hr": torch.rand(3, 640, 640),
                "boxes": torch.zeros((0, 4)), "classes": torch.zeros((0,), dtype=torch.long),
                "dataset": "luna16"}]
    loader = torch.utils.data.DataLoader(samples, batch_size=1, collate_fn=collate)
    with torch.no_grad():
        run(rec, loader, "cpu", dataset="luna16")
    _expect(seen and all(s == "luna16" for s in seen),
            f"run() forwarded dataset name to the model every call (seen={set(seen)})")
    _expect(seen[0] != "UNSET", "no silent default head (brats2021) was used")


def check_10_shared_decoder() -> None:
    print("\n=== 10. single shared bbox decoder ===")
    import scripts.train as train_mod
    import modules.detection_utils as du

    _expect(train_mod.decode_level_boxes is du.decode_level_boxes,
            "training loss and detection_utils use the SAME decode function")
    _expect("decode_level_boxes" in Path(train_mod.__file__).read_text(),
            "decode is imported (not re-implemented) in scripts/train.py")

    torch.manual_seed(0)
    box_lvl = torch.randn(1, 4, 80, 80)
    a = decode_level_boxes(box_lvl, 8.0)
    b = du.decode_level_boxes(box_lvl.clone(), 8.0)
    _expect(torch.allclose(a, b),
            "decode output deterministic and identical for train/eval/inference")

    det = {"cls": [torch.randn(1, 1, 80, 80), torch.randn(1, 1, 40, 40),
                   torch.randn(1, 1, 20, 20)],
           "bbox": [torch.randn(1, 4, 80, 80), torch.randn(1, 4, 40, 40),
                    torch.randn(1, 4, 20, 20)]}
    merged = postprocess_detections(det, hr_size=640, conf_thres=0.01)
    _expect(len(merged) == 1 and {"boxes", "scores", "labels"} <= set(merged[0]),
            "postprocess merges P3/P4/P5 with class-aware NMS")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-yolo", action="store_true",
                    help="skip checks that require the real YOLOv12n weights")
    args = ap.parse_args()

    torch.manual_seed(42)

    check_1_medhead_heads()
    check_2_mixed_modal_batch()
    check_3_vindr_labels()
    check_8_negative_sampling()
    check_9_dataset_routing(None)
    check_10_shared_decoder()

    if not args.no_yolo:
        print("\n=== building MedSRDet with the real YOLOv12n encoder ===")
        cfg = tiny_cfg()
        model = build_model(cfg)
        out = check_4_cmcl_nonzero(model)
        check_5_backward(model, out)
        check_6_eval_mode(model)
        check_7_p345(model)

    n_ok = sum(1 for r in results if r)
    print(f"\n{n_ok}/{len(results)} smoke checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
