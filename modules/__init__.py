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
from .yolo_encoder import YOLOv12FeatureEncoder
from .detection_utils import (
    decode_level_boxes,
    decode_medhead_outputs,
    postprocess_detections,
    compute_map50,
)

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
    "YOLOv12FeatureEncoder",
    "decode_level_boxes",
    "decode_medhead_outputs",
    "postprocess_detections",
    "compute_map50",
]
