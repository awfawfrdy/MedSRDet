"""Self-contained MFASR / CMCL / MedHead modules (reviewer repository).

Mirrors the aligned implementations from the revision work.  These files
depend only on torch / torchvision (no basicsr, no full ultralytics source
tree); the YOLOv12 backbone is pulled from the installed ``ultralytics``
package by the ``medsrdet`` wrapper.
"""
from .mfasrnet_arch import MFASRNet
from .rrdbnet_arch import RRDB, RRDBNet, ResidualDenseBlock
from .loss_mfasr import MFASRLoss, FrequencyLoss, VGGPerceptualLoss
from .cmcl import CMCL
from .medhead import MedHead, ScaleAwareAttention, SpatialAwareAttention, TaskAwareAttention

__all__ = [
    "MFASRNet",
    "RRDB",
    "RRDBNet",
    "ResidualDenseBlock",
    "MFASRLoss",
    "FrequencyLoss",
    "VGGPerceptualLoss",
    "CMCL",
    "MedHead",
    "ScaleAwareAttention",
    "SpatialAwareAttention",
    "TaskAwareAttention",
]
