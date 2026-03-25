"""
Post-hoc saliency visualisation via GradCAM, GradCAM++, and ScoreCAM.

These are used **only during evaluation / inference** to produce interpretable
attention maps.  They do NOT participate in the training loss.

Requires: ``pip install grad-cam``
"""

from __future__ import annotations
from typing import Optional

import torch
import numpy as np

# try:
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
HAS_GRADCAM = True
# except ImportError:
#     HAS_GRADCAM = False


_METHOD_MAP = {}
if HAS_GRADCAM:
    _METHOD_MAP = {
        "gradcam": GradCAM,
        "gradcam_plusplus": GradCAMPlusPlus,
        "score_cam": ScoreCAM,
    }

def get_target_layer(model):
    bt = model.backbone_type
    if bt in ("efficientnet",):
        return [model.backbone[-1]]
    elif bt == "resnet":
        return [model.backbone[-1][-1].conv2]
    elif bt in ("vit", "swin"):
        if hasattr(model.backbone, "blocks"):
            return [model.backbone.blocks[-1].norm1]
        elif hasattr(model.backbone, "layers"):
            return [model.backbone.layers[-1].blocks[-1].norm1]
    elif bt == "medimageinsight":
        # DaViT — hook into the last block's norm
        if hasattr(model.backbone, "blocks"):
            last_stage = model.backbone.blocks[-1]
            if hasattr(last_stage, "norm1"):
                return [last_stage.norm1]
        # Swin-based MedImageInsight fallback
        if hasattr(model.backbone, "layers"):
            return [model.backbone.layers[-1].blocks[-1].norm1]
        return [model.adapter.conv2]  # last resort
    elif bt == "biomedclip":
        if hasattr(model.backbone, "trunk"):
            return [model.backbone.trunk.blocks[-1].norm1]
    return [model.adapter.conv2]


class _ViewWrapper(torch.nn.Module):
    def __init__(self, model, view="cc"):
        super().__init__()
        self.model = model
        self.view = view

    def forward(self, x):
        # Compute features for ONLY this view, skip fusion
        sam_feats, global_pool = self.model.forward_view(x)
        
        if self.model.pooling_mode == "sam_roi":
            # Use global pool doubled as proxy (no GT mask available here)
            feat = torch.cat([global_pool, global_pool], dim=1)
        elif self.model.pooling_mode == "learned":
            roi = self.model.saliency_head(sam_feats)
            roi_pool = (sam_feats * roi).sum(dim=(2,3)) / (roi.sum(dim=(2,3)) + 1e-6)
            feat = torch.cat([global_pool, roi_pool], dim=1)
        else:
            feat = global_pool

        # Dummy age
        B = x.shape[0]
        age_emb = self.model.age_encoder(torch.zeros(B, 1, device=x.device))
        combined = torch.cat([feat, age_emb], dim=1)
        return self.model.head_severity(combined)

def compute_saliency_map(
    model,
    image: torch.Tensor,
    method: str = "gradcam",
    target_class: Optional[int] = None,
    view: str = "cc",
) -> Optional[np.ndarray]:
    """Generate a saliency map for a single image.

    Parameters
    ----------
    model : MammoSightModel
    image : ``[1, 3, H, W]`` tensor (normalised).
    method : ``"gradcam"`` | ``"gradcam_plusplus"`` | ``"score_cam"``
    target_class : int or None (uses argmax if None).
    view : ``"cc"`` or ``"mlo"``.

    Returns
    -------
    cam : ``[H, W]`` numpy array in ``[0, 1]``, or ``None`` if unavailable.
    """
    if not HAS_GRADCAM:
        print("[saliency] grad-cam not installed.  pip install grad-cam")
        return None
    if method == "none" or method not in _METHOD_MAP:
        return None

    wrapper = _ViewWrapper(model, view=view)
    target_layers = get_target_layer(model)
    cam_cls = _METHOD_MAP[method]

    with cam_cls(model=wrapper, target_layers=target_layers) as cam:
        targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
        grayscale_cam = cam(input_tensor=image, targets=targets)
    return grayscale_cam[0]  # [H, W]
