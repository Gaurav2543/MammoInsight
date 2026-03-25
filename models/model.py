"""
MammoFormer – Multi-task Mammography Foundation Model.

Pluggable components
--------------------
**Backbone**        EfficientNet · ResNet · ViT · Swin · BiomedCLIP · MedImageInsight
**Pooling**         global · sam_roi · learned (spatial attention)
**Fusion**          cross_attention (proposed) · concat (baseline)
**Segmentation**    SAM-Med2D decoder (optional)
**Task heads**      Severity · Density · BI-RADS (ordinal) | Abnormality · Molecular (nominal)
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ── optional imports (fail gracefully) ───────────────────────────────────
# try:
import timm
# except ImportError:
#     timm = None

# try:
import open_clip
# except ImportError:
#     open_clip = None

# try:
from src.sam_med2d.mask_decoder import MaskDecoder
from src.sam_med2d.transformer import TwoWayTransformer
# from src.mammo_clip.image_encoder import EfficientNet_Mammo

# except ImportError:
#     MaskDecoder = None
#     TwoWayTransformer = None

# try:
from src.medimageinsights.medimageinsightmodel import MedImageInsight
# except ImportError:
#     MedImageInsight = None


# =========================================================================
# Building blocks
# =========================================================================

class UniversalAdapter(nn.Module):
    """Project any backbone feature map to ``[B, 256, 64, 64]`` for SAM."""

    def __init__(self, in_channels: int, target_res: int = 64):
        super().__init__()
        self.target_res = target_res
        self.conv1 = nn.Conv2d(in_channels, 256, 1)
        self.gn = nn.GroupNorm(32, 256)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ViT-style [B, N, C] → [B, C, H, W]
        if x.dim() == 3:
            B, N, C = x.shape
            s = int(math.sqrt(N))
            x = x.transpose(1, 2).view(B, C, s, s)
        x = self.act(self.gn(self.conv1(x)))
        x = self.conv2(x)
        if x.shape[-1] != self.target_res:   # Resize to target resolution for SAM
            x = F.interpolate(x, size=(self.target_res, self.target_res),
                              mode="bilinear", align_corners=False)
        return x


class SaliencyBlock(nn.Module):
    """Lightweight learned spatial-attention map (sigmoid output)."""

    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [B, 1, H, W]


class CrossViewFusion(nn.Module):
    """Attention-based fusion of CC and MLO views (proposed)."""

    def __init__(self, dim: int = 512):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())

    def forward(self, x_cc, x_mlo, view_mask):
        seq = torch.stack([x_cc, x_mlo], dim=1)          # [B, 2, D]
        pad_mask = view_mask == 0                          # True → padded
        if pad_mask.all():
            return torch.zeros_like(x_cc)
        attn_out, _ = self.attn(seq, seq, seq, key_padding_mask=pad_mask)
        seq = self.norm(seq + attn_out)
        cc_out, mlo_out = seq[:, 0], seq[:, 1]
        g = self.gate(torch.cat([cc_out, mlo_out], -1))
        fused = g * cc_out + (1 - g) * mlo_out
        # Handle missing views
        has_cc = view_mask[:, 0:1]
        has_mlo = view_mask[:, 1:2]
        return torch.where(
            has_cc * has_mlo > 0.5, fused,
            cc_out * has_cc + mlo_out * has_mlo,
        )


class ConcatFusion(nn.Module):
    """Simple concatenation + linear projection baseline."""

    def __init__(self, dim: int = 512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

    def forward(self, x_cc, x_mlo, view_mask):
        has_cc = view_mask[:, 0:1]
        has_mlo = view_mask[:, 1:2]
        # zero-out missing views
        x_cc = x_cc * has_cc
        x_mlo = x_mlo * has_mlo
        return self.proj(torch.cat([x_cc, x_mlo], dim=-1))


# =========================================================================
# Main model
# =========================================================================

class MammoSightModel(nn.Module):
    """
    Parameters
    ----------
    backbone_type : str
        ``"efficientnet"`` | ``"resnet"`` | ``"vit"`` | ``"swin"`` |
        ``"biomedclip"`` | ``"medimageinsight"``
    backbone_name : str
        Variant / HuggingFace ID / local path.
    sam_checkpoint_path : str or None
        Path to SAM-Med2D weights.  ``None`` → segmentation disabled.
    pooling_mode : str
        ``"global"`` | ``"sam_roi"`` | ``"learned"``
    fusion_mode : str
        ``"cross_attention"`` | ``"concat"``
    """

    def __init__(
        self,
        backbone_type: str = "efficientnet",
        backbone_name: str = "efficientnet_b5",
        # sam_checkpoint_path: str | None = None,
        sam_checkpoint_path= None,
        pooling_mode: str = "learned",
        fusion_mode: str = "cross_attention",
    ):
        super().__init__()
        self.backbone_type = backbone_type.lower()
        self.pooling_mode = pooling_mode.lower()
        self.fusion_mode = fusion_mode.lower()

        # ── 1. Backbone ──────────────────────────────────────────────────
        self.embed_dim = self._init_backbone(backbone_name)
        print(f"[model] backbone={self.backbone_type}  embed_dim={self.embed_dim}  "
              f"pooling={self.pooling_mode}  fusion={self.fusion_mode}")

        # ── 2. Adapter ───────────────────────────────────────────────────
        self.adapter = UniversalAdapter(self.embed_dim)

        # ── 3. Age encoder ───────────────────────────────────────────────
        self.age_encoder = nn.Sequential(
            nn.Linear(1, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 256), nn.LayerNorm(256),
        )

        # ── 4. Segmentation decoder (SAM) ────────────────────────────────
        self.sam_decoder = None
        if MaskDecoder is not None and TwoWayTransformer is not None:
            self.sam_transformer = TwoWayTransformer(
                depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8,
            )
            self.sam_decoder = MaskDecoder(
                transformer_dim=256, transformer=self.sam_transformer,
                num_multimask_outputs=3, activation=nn.GELU,
                iou_head_depth=3, iou_head_hidden_dim=256,
            )
            if sam_checkpoint_path:
                self._load_sam_weights(sam_checkpoint_path)
        else:
            print("[model] SAM decoder disabled (missing imports)")

        # ── 5. Pooling ───────────────────────────────────────────────────
        self.saliency_head = None
        if self.pooling_mode == "learned":
            self.saliency_head = SaliencyBlock(256)

        # Feature dim after pooling: global (256) or global+roi (512)
        view_dim = 512 if self.pooling_mode != "global" else 256

        # ── 6. Fusion ────────────────────────────────────────────────────
        if self.fusion_mode == "cross_attention":
            self.fusion = CrossViewFusion(dim=view_dim)
        else:
            self.fusion = ConcatFusion(dim=view_dim)

        # ── 7. Classification heads ──────────────────────────────────────
        # Input: view_dim (fusion) + 256 (age)
        head_in = view_dim + 256
        self.head_severity = nn.Linear(head_in, 2)      # 3 classes → 2 cutpoints
        self.head_density = nn.Linear(head_in, 3)       # 4 classes → 3 cutpoints
        self.head_birads = nn.Linear(head_in, 5)        # 6 classes → 5 cutpoints
        self.head_abnormality = nn.Linear(head_in, 3)   # nominal
        self.head_molecular = nn.Linear(head_in, 4)     # nominal

    # ─── backbone initialisation ─────────────────────────────────────────

    def _init_backbone(self, name: str) -> int:
        bt = self.backbone_type
        if bt == "efficientnet":
            return self._init_efficientnet(name)
        elif bt == "resnet":
            return self._init_resnet(name)
        elif bt == "vit":
            return self._init_vit(name)
        elif bt == "swin":
            return self._init_swin(name)
        elif bt == "biomedclip":
            return self._init_biomedclip(name)
        elif bt == "medimageinsight":
            return self._init_medimageinsight(name)
        elif bt == "mammoclip":
            return self._init_mammoclip(name)
        raise ValueError(f"Unknown backbone: {bt}")

    def _init_efficientnet(self, name):
        lookup = {
            "efficientnet_b0": (models.efficientnet_b0, 1280),
            "efficientnet_b2": (models.efficientnet_b2, 1408),
            "efficientnet_b5": (models.efficientnet_b5, 2048),
            "efficientnet_b7": (models.efficientnet_b7, 2560),
        }
        key = name.lower().replace("-", "_")
        for k, (fn, dim) in lookup.items():
            if k in key:
                self.backbone = fn(weights="DEFAULT").features
                return dim
        raise ValueError(f"Unknown EfficientNet variant: {name}")

    def _init_resnet(self, name):
        lookup = {
            "34": (models.resnet34, 512),
            "50": (models.resnet50, 2048),
            "101": (models.resnet101, 2048),
        }
        for tag, (fn, dim) in lookup.items():
            if tag in name:
                base = fn(weights="DEFAULT")
                self.backbone = nn.Sequential(*list(base.children())[:-2])
                return dim
        raise ValueError(f"Unknown ResNet variant: {name}")

    def _init_vit(self, name):
        # if timm is None:
        #     raise ImportError("timm is required for ViT.  pip install timm")
        self.backbone = timm.create_model(name, pretrained=True, num_classes=0)
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = self.backbone.forward_features(dummy)
        dim = out.shape[-1]
        return dim

    def _init_swin(self, name):
        # if timm is None:
        #     raise ImportError("timm is required for Swin.  pip install timm")
        self.backbone = timm.create_model(name, pretrained=True, num_classes=0)
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = self.backbone.forward_features(dummy)
        dim = out.shape[-1]
        return dim

    def _init_biomedclip(self, name):
        # if open_clip is None:
        #     raise ImportError("open_clip required.  pip install open-clip-torch")
        model_id = name or "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        model, _, _ = open_clip.create_model_and_transforms("hf-hub:" + model_id)
        self.backbone = model.visual
        return 768

    def _init_medimageinsight(self, name):
        # if MedImageInsight is None:
        #     raise ImportError("MedImageInsight not found")
        wrapper = MedImageInsight(
            model_dir=name,
            vision_model_name="medimageinsigt-v1.0.0.pt",
            language_model_name="language_model.pth",
        )
        wrapper.load_model()
        self.backbone = wrapper.model.image_encoder
        dim = getattr(self.backbone, "dim_out",
                      getattr(self.backbone, "num_features", 768))
        del wrapper.model.lang_encoder, wrapper.model.lang_projection, wrapper
        return dim
    
    def _init_mammoclip(self, name):
        # Use Mammo-CLIP's EfficientNet_Mammo architecture (no VLM weights, just ImageNet init)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

        encoder = EfficientNet_Mammo(
            name=name,
            pretrained=True,   
            in_chans=3,
        )
        embed_dim = encoder.out_dim
        self.backbone = encoder
        self.backbone_type = "mammoclip"
        return embed_dim

    # ─── SAM weight loading ──────────────────────────────────────────────

    def _load_sam_weights(self, path):
        ckpt = torch.load(path, map_location="cpu")
        sd = ckpt.get("model", ckpt)
        local_keys = set(self.sam_decoder.state_dict().keys())
        filtered = {
            k.replace("mask_decoder.", ""): v
            for k, v in sd.items()
            if k.replace("mask_decoder.", "") in local_keys
        }
        self.sam_decoder.load_state_dict(filtered, strict=False)
        print(f"[model] Loaded SAM decoder weights ({len(filtered)} keys)")

    # ─── per-view feature extraction ─────────────────────────────────────

    def forward_view(self, img: torch.Tensor):
        """Return ``(sam_feats [B,256,64,64], global_pool [B,256])``."""
        bt = self.backbone_type

        if bt == "biomedclip":
            if hasattr(self.backbone, "forward_features"):
                x = self.backbone.forward_features(img)
                raw = x[:, 1:] if x.shape[1] == 197 else x
            elif hasattr(self.backbone, "trunk"):
                raw = self.backbone.trunk.forward_features(img)[:, 1:]
            else:
                raise AttributeError("Unknown BiomedCLIP structure")
        elif bt == "medimageinsight":
            cls_name = self.backbone.__class__.__name__
            if "DaViT" in cls_name:
                x, sz = img, (img.size(2), img.size(3))
                for conv, blk in zip(self.backbone.convs, self.backbone.blocks):
                    x, sz = conv(x, sz)
                    x, sz = blk(x, sz)
                raw = x
            elif "Swin" in cls_name:
                x = self.backbone.patch_embed(img)
                if self.backbone.ape:
                    x = x + self.backbone.absolute_pos_embed
                x = self.backbone.pos_drop(x)
                for layer in self.backbone.layers:
                    x = layer(x)
                raw = self.backbone.norm(x)
            else:
                raw = self.backbone.forward_features(img)
        elif bt in ("vit", "swin"):
            raw = self.backbone.forward_features(img)  # timm API
            # ViT: [B, 197, C] → remove CLS → [B, 196, C]
            if raw.dim() == 3 and raw.shape[1] == 197:
                raw = raw[:, 1:]
            # Swin: [B, H, W, C] → [B, C, H, W]
            if raw.dim() == 4 and raw.shape[-1] != raw.shape[1]:
                raw = raw.permute(0, 3, 1, 2)        
        elif bt == "mammoclip":
            raw = self.backbone.model.forward_features(img)  # [B, C, H, W] from timm
        else:
            # CNN (EfficientNet / ResNet)
            raw = self.backbone(img)
            

        sam_feats = self.adapter(raw)             # [B, 256, 64, 64]
        global_pool = sam_feats.mean(dim=(2, 3))  # [B, 256]
        return sam_feats, global_pool

    # ─── forward ─────────────────────────────────────────────────────────

    def forward(self, batch: dict, return_features: bool = False) -> dict:
        cc_sam, cc_pool = self.forward_view(batch["cc_image"])
        mlo_sam, mlo_pool = self.forward_view(batch["mlo_image"])

        # ── segmentation ─────────────────────────────────────────────────
        mask_cc, mask_mlo = None, None
        if self.sam_decoder is not None:
            B = cc_sam.shape[0]
            sparse = torch.empty(B, 0, 256, device=cc_sam.device)
            dense = torch.zeros(B, 256, 64, 64, device=cc_sam.device)
            pos = torch.zeros_like(cc_sam)
            mask_cc, _ = self.sam_decoder(cc_sam, pos, sparse, dense, False)
            mask_mlo, _ = self.sam_decoder(mlo_sam, pos, sparse, dense, False)

        # ── pooling ──────────────────────────────────────────────────────
        if self.pooling_mode == "global":
            cc_feat, mlo_feat = cc_pool, mlo_pool

        elif self.pooling_mode == "sam_roi":
            cc_feat, mlo_feat = self._sam_roi_pool(
                cc_sam, mlo_sam, cc_pool, mlo_pool, mask_cc, mask_mlo, batch
            )

        elif self.pooling_mode == "learned":
            roi_cc = self.saliency_head(cc_sam)
            roi_mlo = self.saliency_head(mlo_sam)
            cc_roi_p = (cc_sam * roi_cc).sum(dim=(2, 3)) / (roi_cc.sum(dim=(2, 3)) + 1e-6)
            mlo_roi_p = (mlo_sam * roi_mlo).sum(dim=(2, 3)) / (roi_mlo.sum(dim=(2, 3)) + 1e-6)
            cc_feat = torch.cat([cc_pool, cc_roi_p], dim=1)
            mlo_feat = torch.cat([mlo_pool, mlo_roi_p], dim=1)

        else:
            raise ValueError(f"Unknown pooling_mode: {self.pooling_mode}")

        # ── fusion ───────────────────────────────────────────────────────
        fused = self.fusion(cc_feat, mlo_feat, batch["view_mask"])

        # ── metadata ─────────────────────────────────────────────────────
        age = batch["labels"]["age"].float().unsqueeze(1)
        age_emb = self.age_encoder(age)
        combined = torch.cat([fused, age_emb], dim=1)

        # ── task heads ───────────────────────────────────────────────────
        out = {
            "classification": self.head_severity(combined),
            "density": self.head_density(combined),
            "birads": self.head_birads(combined),
            "abnormality": self.head_abnormality(combined),
            "molecular": self.head_molecular(combined),
            "pred_mask_cc": mask_cc,
            "pred_mask_mlo": mask_mlo,
        }
        if return_features:
            out["combined_features"] = combined
        return out

    # ─── SAM-based RoI pooling ───────────────────────────────────────────

    def _sam_roi_pool(self, cc_sam, mlo_sam, cc_pool, mlo_pool,
                      mask_cc, mask_mlo, batch):
        """Use GT masks during training (teacher forcing) else predictions."""
        if mask_cc is None:
            # SAM disabled → fall back to global + duplicate
            return (
                torch.cat([cc_pool, cc_pool], dim=1),
                torch.cat([mlo_pool, mlo_pool], dim=1),
            )

        if self.training and batch["has_seg"].sum() > 0:
            roi_cc = F.interpolate(batch["cc_mask"], (64, 64), mode="nearest")
            roi_mlo = F.interpolate(batch["mlo_mask"], (64, 64), mode="nearest")
        else:
            roi_cc = (torch.sigmoid(mask_cc) > 0.5).float()
            roi_mlo = (torch.sigmoid(mask_mlo) > 0.5).float()
            roi_cc = F.interpolate(roi_cc, (64, 64), mode="nearest")
            roi_mlo = F.interpolate(roi_mlo, (64, 64), mode="nearest")

        def _weighted_pool(feats, roi, glob):
            s = roi.sum(dim=(2, 3))
            weighted = (feats * roi).sum(dim=(2, 3)) / (s + 1e-6)
            has = (s > 10).float()
            pooled = has * weighted + (1 - has) * glob
            return torch.cat([glob, pooled], dim=1)

        return (
            _weighted_pool(cc_sam, roi_cc, cc_pool),
            _weighted_pool(mlo_sam, roi_mlo, mlo_pool),
        )
