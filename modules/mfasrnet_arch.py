"""MFASR network — aligned implementation.

MFASR (Medical Frequency-Aware Super-Resolution), as described in the revised
manuscript (Methods 3.2 / Tables 2-4).

Architecture (manuscript-explicit):
    - Shallow feature extraction: Conv2d 3x3, 3 -> 64 channels.
    - Deep feature extraction: RRDB trunk of 23 RRDB blocks; each RRDB contains
      3 Residual Dense Blocks; growth channels gc = 32; local residual scaling
      beta = 0.2; feature channels stay 64; long skip connection.
    - Upsampling (total x4): two stages, each "nearest-neighbor x2 -> 3x3 conv".
    - Reconstruction: final 3x3 Conv2d 64 -> 3.

The RRDB / RDB building blocks are reused verbatim from ``rrdbnet_arch.py``
(the standard ESRGAN RRDB implementation already present in this project);
no new residual architecture is invented here.
"""

import torch
import torch.nn as nn

from .rrdbnet_arch import RRDBNet  # reuse existing correct RRDB trunk

# Manuscript Methods 3.2 constants (Tables 2-4)
_NUM_BLOCKS = 23       # RRDB blocks in the trunk
_NUM_GROW_CH = 32      # growth channels gc
_NUM_FEAT = 64         # feature channels
_SCALE = 4             # total upsampling factor
_BETA = 0.2            # local residual scaling (already used inside RRDBNet)


class MFASRNet(RRDBNet):
    """MFASR reconstruction network.

    Inherits the full RRDB-based trunk and the two-stage nearest-neighbour
    upsampling from ``RRDBNet`` so the forward path is exactly:

        x[B,3,H,W] -> conv_first(3->64) -> 23xRRDB trunk -> long skip
        -> nearest x2 + 3x3 conv -> nearest x2 + 3x3 conv
        -> conv_last(64->3) -> y[B,3,4H,4W]

    The manuscript does not define a discriminator / GAN objective for MFASR;
    therefore this module is purely a reconstruction network.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = _NUM_FEAT,
        num_block: int = _NUM_BLOCKS,
        num_grow_ch: int = _NUM_GROW_CH,
        scale: int = _SCALE,
        **kwargs,
    ) -> None:
        super().__init__(
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
            scale=scale,
            num_feat=num_feat,
            num_block=num_block,
            num_grow_ch=num_grow_ch,
        )


if __name__ == "__main__":
    # Quick shape sanity check (not part of the project test suite)
    model = MFASRNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    print("input :", x.shape)
    print("output:", y.shape)
    assert tuple(y.shape) == (1, 3, 256, 256), f"unexpected output shape {tuple(y.shape)}"
    print("MFASRNet x4 shape check passed")
