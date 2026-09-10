"""Configuration / shape audit of the three MedSRDet modules.

Runs without any dataset and checks the released implementation against the
manuscript tables:

    MFASR   -> Methods 3.2, Table 2
    CMCL    -> Methods 3.3, Table 3, Appendix B (Table B.1)
    MedHead -> Methods 3.4, Table 4, Appendix B (Table B.1)

Usage
-----
    python scripts/selfcheck.py                # skip the VGG perceptual term
    python scripts/selfcheck.py --vgg          # also build the VGG-19 loss
                                               # (downloads ImageNet weights)

Exit code is 0 when every check passes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import (  # noqa: E402
    CMCL,
    MFASRLoss,
    MFASRNet,
    MedHead,
)

OK = "  [ok]"
FAIL = "  [FAIL]"


def _expect(cond: bool, msg: str, results: list) -> None:
    print(f"{OK if cond else FAIL} {msg}")
    results.append(cond)


# --------------------------------------------------------------------------- #
# MFASR — Methods 3.2 / Table 2
# --------------------------------------------------------------------------- #
def check_mfasr(results: list) -> None:
    print("\n=== MFASR (Methods 3.2, Table 2) ===")
    net = MFASRNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                   num_block=23, num_grow_ch=32, scale=4)

    n_rrdb = len(net.body)
    _expect(n_rrdb == 23, f"RRDB trunk contains 23 blocks (got {n_rrdb})", results)

    first = net.body[0]
    n_rdb = sum(1 for name, _ in first.named_children() if name.startswith("rdb"))
    _expect(n_rdb == 3, f"each RRDB contains 3 RDBs (got {n_rdb})", results)
    _expect(first.rdb1.conv1.out_channels == 32,
            f"growth channels gc = 32 (got {first.rdb1.conv1.out_channels})", results)

    _expect(net.conv_first.in_channels == 3 and net.conv_first.out_channels == 64,
            f"shallow feature extraction 3->64 "
            f"(got {net.conv_first.in_channels}->{net.conv_first.out_channels})", results)
    _expect(net.conv_last.out_channels == 3, "reconstruction layer outputs 3 channels", results)

    # CMCL lesion view: 224 x 224 patch -> bicubic 56 x 56 -> MFASR x4 -> 224 x 224
    x = torch.randn(2, 3, 56, 56)
    y = net(x)
    _expect(tuple(y.shape) == (2, 3, 224, 224),
            f"CMCL lesion view 56x56 -> 224x224 (got {tuple(y.shape)})", results)

    # Main detection pathway: HR 640 x 640 <- LR 160 x 160 (Appendix D.1)
    x = torch.randn(1, 3, 160, 160)
    y = net(x)
    _expect(tuple(y.shape) == (1, 3, 640, 640),
            f"detection pathway 160x160 -> 640x640 (got {tuple(y.shape)})", results)


def check_mfasr_loss(results: list, with_vgg: bool) -> None:
    print("\n=== MFASR loss (Eq. 1, Table 2) ===")
    if with_vgg:
        crit = MFASRLoss()
    else:
        # Build the loss without the VGG branch to keep the check offline.
        crit = MFASRLoss(lambda_perc=0.0)

    _expect(crit.lambda_pix == 1.0 and crit.lambda_freq == 0.1
            and (crit.lambda_perc == 0.05 or not with_vgg),
            f"L_SR weights = {crit.lambda_pix} / {crit.lambda_freq} / "
            f"{crit.lambda_perc}  (L_pix / 0.1 L_freq / 0.05 L_perc)", results)

    sr = torch.rand(2, 3, 32, 32)
    hr = torch.rand(2, 3, 32, 32)
    _, parts = crit(sr, hr)
    _expect(set(parts) >= {"l_pix", "l_freq", "l_total"},
            f"loss returns the component breakdown {sorted(parts)}", results)
    _expect(parts["l_freq"] >= 0, "frequency term is a non-negative magnitude distance", results)

    # Magnitude-only supervision: phase is not supervised.
    fft_sr = torch.fft.fft2(sr, norm="ortho")
    manual = (torch.abs(fft_sr) - torch.abs(torch.fft.fft2(hr, norm="ortho"))).abs().mean()
    _expect(abs(manual.item() - parts["l_freq"]) < 1e-5,
            "L_freq equals the L1 distance of the FFT magnitude spectra", results)

    if with_vgg:
        _expect(crit.perc_loss is not None and not crit.perc_loss.features.training,
                "VGG-19 perceptual branch is frozen and in eval mode", results)
    else:
        print("  [--] VGG perceptual branch not built (rerun with --vgg)")


# --------------------------------------------------------------------------- #
# CMCL — Methods 3.3 / Table 3 / Appendix B
# --------------------------------------------------------------------------- #
def check_cmcl(results: list) -> None:
    print("\n=== CMCL (Methods 3.3, Table 3, Appendix B) ===")
    # Backbone channel widths of P3/P4/P5 are dataset/model dependent; any
    # triple works here because the per-scale 1x1 projections are built from it.
    in_channels = (64, 128, 256)
    cmcl = CMCL(in_channels=in_channels, feature_dim=256)
    cmcl.train()

    _expect(cmcl.temperature == 0.07, f"temperature tau = 0.07 (got {cmcl.temperature})", results)
    _expect(cmcl.lambda_reg == 1e-4, f"lambda_reg = 1e-4 (got {cmcl.lambda_reg})", results)
    _expect(cmcl.momentum == 0.999, f"momentum m = 0.999 (got {cmcl.momentum})", results)
    _expect(tuple(cmcl.queue.shape) == (65536, 128),
            f"memory queue K = 65,536 x 128 (got {tuple(cmcl.queue.shape)})", results)

    n_proj = len(cmcl.scale_proj) if cmcl.scale_proj is not None else 0
    _expect(n_proj == 3,
            f"one dedicated 1x1 projection per scale P3/P4/P5 (got {n_proj})", results)

    # Appendix B: P3/P4/P5 -> per-scale 1x1 -> GAP -> element-wise mean -> 256
    feats = [torch.randn(4, c, 28, 28) for c in in_channels]
    z = cmcl._embed_query(feats)
    _expect(tuple(z.shape) == (4, 128),
            f"multi-scale aggregation -> 128-d embedding (got {tuple(z.shape)})", results)
    _expect(bool(torch.allclose(z.norm(dim=1), torch.ones(4), atol=1e-5)),
            "embeddings are l2-normalized (sim(u,v) = u^T v)", results)

    # Appendix B: "fused by element-wise averaging".  Verified on a projection
    # free probe so the expected value is computable in closed form.
    probe = CMCL(in_channels=None, feature_dim=256)
    a, b, c = (torch.randn(4, 256, 14, 14) for _ in range(3))
    gap = lambda t: torch.nn.functional.adaptive_avg_pool2d(t, (1, 1)).flatten(1)
    expected_mean = (gap(a) + gap(b) + gap(c)) / 3.0
    got = probe._to_vector([a, b, c])
    _expect(bool(torch.allclose(got, expected_mean, atol=1e-6)),
            "P3/P4/P5 are fused by element-wise mean (not concat / max)", results)

    # A single-scale tensor is still accepted (backwards compatibility).
    z_single = cmcl._embed_query(feats[0])
    _expect(tuple(z_single.shape) == (4, 128), "single-scale fallback still works", results)

    # Positive pair = (augmented view_i, MFASR view_i); negatives = in-batch + queue.
    loss = cmcl(feats, [f.clone() for f in feats])
    _expect(torch.is_tensor(loss) and loss.ndim == 0,
            "forward() returns a scalar InfoNCE + regularization loss", results)

    # Inference must not depend on CMCL.
    cmcl.eval()
    with torch.no_grad():
        out = cmcl(feats, feats)
    _expect(float(out) == 0.0, "eval() mode returns a zero scalar (training-only)", results)
    cmcl.train()


# --------------------------------------------------------------------------- #
# MedHead — Methods 3.4 / Table 4 / Appendix B
# --------------------------------------------------------------------------- #
def check_medhead(results: list) -> None:
    print("\n=== MedHead (Methods 3.4, Table 4, Appendix B) ===")
    in_channels = (128, 256, 512)
    head = MedHead(in_channels=in_channels, nc=1)
    head.train()

    _expect(head.gamma == 0.2, f"residual scaling gamma = 0.2 (got {head.gamma})", results)
    _expect(head.channels == 256, "unified channel dimension is 256", results)

    n_proj = len(head.projection)
    _expect(n_proj == 3,
            f"P3/P4/P5 are independently projected to 256 (got {n_proj} projections)", results)
    _expect(all(p.out_channels == 256 for p in head.projection),
            "each scale projection outputs 256 channels", results)

    # scale-aware branch: GAP -> MLP 256->64->256 -> Sigmoid
    mlp = head.scale_attn.mlp
    _expect(mlp[0].in_features == 256, "scale-aware MLP input is 256", results)
    _expect(mlp[0].out_features == 64 and mlp[2].out_features == 256,
            "scale-aware MLP is 256 -> 64 -> 256", results)

    # spatial-aware branch: 3x3 deformable conv, groups = 8
    _expect(head.spatial_attn.deform_conv.kernel_size == (3, 3)
            and head.spatial_attn.groups == 8,
            "spatial-aware branch is a 3x3 deformable conv with groups = 8", results)

    # three parallel branches + fusion
    _expect(len(head.ctx_conv) == 3, "one lightweight 3x3 conv per parallel branch", results)
    _expect(head.fusion_conv.in_channels == 256 * 3
            and head.fusion_conv.out_channels == 256,
            "fusion is concat(3x256) -> 1x1 conv -> 256", results)
    _expect(head.reg_head.out_channels == 4,
            "regression head outputs (dx, dy, dw, dh)", results)

    feats = [torch.randn(2, c, s, s) for c, s in zip(in_channels, (80, 40, 20))]
    out = head(feats)
    _expect(len(out["features"]) == 3, "three refined feature levels are returned", results)
    _expect(all(f.shape[1] == 256 for f in out["features"]),
            "every refined level has 256 channels", results)
    _expect(tuple(out["cls"][0].shape) == (2, 1, 80, 80),
            f"classification output shape (got {tuple(out['cls'][0].shape)})", results)
    _expect(tuple(out["bbox"][0].shape) == (2, 4, 80, 80),
            f"regression output shape (got {tuple(out['bbox'][0].shape)})", results)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vgg", action="store_true",
                    help="also build the VGG-19 perceptual loss "
                         "(requires downloading ImageNet weights)")
    args = ap.parse_args()

    torch.manual_seed(42)
    results: list = []

    check_mfasr(results)
    check_mfasr_loss(results, with_vgg=args.vgg)
    check_cmcl(results)
    check_medhead(results)

    n_ok = sum(1 for r in results if r)
    print(f"\n{n_ok}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
