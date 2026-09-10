"""MFASR reconstruction losses — aligned implementation.

Losses follow the revised manuscript (Methods 3.2, Eq. for L_SR):

    L_SR = L_pix + 0.1 * L_freq + 0.05 * L_perc

Component definitions (manuscript-explicit):
    - L_pix   : L1 loss between I_SR and I_HR           (nn.L1Loss)
    - L_freq  : 2D FFT magnitude-spectrum L1 distance   (magnitude only,
                no phase loss; no adversarial loss)
    - L_perc  : L1 feature distance on pretrained VGG-19 ReLU5_4 features
                (VGG frozen, eval mode)

Unspecified implementation details (documented):
    - VGG19 input normalization uses the torchvision pretrained-weights
      standard mean/std (ImageNet). The manuscript does not state the
      normalization explicitly; torchvision standard handling is applied.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import vgg19, VGG19_Weights
    _HAS_TORCHVISION = True
except Exception:  # pragma: no cover - environment without torchvision
    _HAS_TORCHVISION = False

# Manuscript constants (Methods 3.2)
LAMBDA_PIX_DEFAULT = 1.0
LAMBDA_FREQ_DEFAULT = 0.1
LAMBDA_PERC_DEFAULT = 0.05
# torchvision VGG19 features children are 0..36 (Conv/ReLU/MaxPool interleaved);
# ReLU5_4 is the ReLU at index 35, i.e. slicing features[:36] keeps up to ReLU5_4.
VGG_RELU5_4_SLICE = 36


class FrequencyLoss(nn.Module):
    """Frequency-domain loss: L1 distance between FFT magnitude spectra.

    Only the magnitude is supervised (abs(fft2(...))); no phase term is added,
    matching the manuscript's frequency-loss definition.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fft = torch.fft.fft2(pred, norm="ortho")
        target_fft = torch.fft.fft2(target, norm="ortho")
        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)
        return F.l1_loss(pred_amp, target_amp)


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss on pretrained VGG-19 ReLU5_4 features (L1 distance).

    The VGG feature extractor is frozen (requires_grad=False) and kept in eval
    mode. Inputs are normalized with the torchvision pretrained-weights
    standard mean/std before being fed to the network.
    """

    def __init__(self):
        super().__init__()
        if not _HAS_TORCHVISION:
            raise ImportError("torchvision is required for the VGG perceptual loss.")

        # The manuscript explicitly requires a PRETRAINED VGG-19 (ReLU5_4).
        # Loading random weights (weights=None) is therefore NOT allowed: if
        # the ImageNet-1K V1 checkpoint cannot be loaded we fail loudly instead
        # of silently degrading to a randomly-initialized network.
        try:
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        except Exception as exc:  # pragma: no cover - network / cache failure
            raise RuntimeError(
                "VGG19 perceptual loss requires pretrained IMAGENET1K_V1 "
                "weights (manuscript Methods 3.2). Loading failed; random-"
                "weight fallback is intentionally disabled. "
                f"Original error: {exc}"
            ) from exc
        # Tag recording which checkpoint was actually loaded (used by tests to
        # assert that no weights=None fallback occurred).
        self.pretrained_weights = VGG19_Weights.IMAGENET1K_V1

        # Keep children up to and including ReLU5_4 (index 35 => slice 36).
        self.features = nn.Sequential(*list(vgg.features.children())[:VGG_RELU5_4_SLICE])
        self.features.eval()
        for p in self.features.parameters():
            p.requires_grad = False

        # torchvision standard normalization for pretrained VGG19.
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.features = self.features.to(pred.device)
        self.mean = self.mean.to(pred.device)
        self.std = self.std.to(pred.device)

        pred = self.normalize(pred)
        target = self.normalize(target)
        feat_pred = self.features(pred)
        feat_target = self.features(target)
        return F.l1_loss(feat_pred, feat_target)


class MFASRLoss(nn.Module):
    """Combined MFASR reconstruction loss.

    L_SR = lambda_pix * L_pix + lambda_freq * L_freq + lambda_perc * L_perc

    Manuscript defaults (main experiment): 1.0, 0.1, 0.05.

    Returns ``(total, metrics_dict)`` to keep compatibility with existing
    training scripts that consume ``loss, loss_dict = criterion(sr, hr)``.
    """

    def __init__(self, lambda_pix: float = LAMBDA_PIX_DEFAULT,
                 lambda_freq: float = LAMBDA_FREQ_DEFAULT,
                 lambda_perc: float = LAMBDA_PERC_DEFAULT):
        super().__init__()
        self.lambda_pix = lambda_pix
        self.lambda_freq = lambda_freq
        self.lambda_perc = lambda_perc

        # L_pix: L1 loss (manuscript-explicit; Charbonnier is intentionally
        # not used here).
        self.pix_loss = nn.L1Loss()
        self.freq_loss = FrequencyLoss()
        self.perc_loss = VGGPerceptualLoss() if lambda_perc > 0 else None

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        l_pix = self.pix_loss(pred, target)
        l_freq = self.freq_loss(pred, target)

        if self.perc_loss is not None:
            l_perc = self.perc_loss(pred, target)
        else:
            l_perc = torch.tensor(0.0, device=pred.device)

        total = (
            self.lambda_pix * l_pix
            + self.lambda_freq * l_freq
            + self.lambda_perc * l_perc
        )

        return total, {
            "l_pix": l_pix.item(),
            "l_freq": l_freq.item(),
            "l_perc": l_perc.item() if torch.is_tensor(l_perc) else float(l_perc),
            "l_total": total.item(),
        }
