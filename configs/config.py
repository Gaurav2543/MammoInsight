"""
Central configuration for MammoFormer.

All architecture choices, training hyperparameters, and feature flags live here.
Modify this file or override via CLI / W&B sweep to control every aspect of the
training pipeline.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class Config:
    """Master configuration for MammoFormer training."""

    # ── Project / Logging ────────────────────────────────────────────────
    PROJECT_NAME: str = "MammoSight"
    RUN_NAME: str = "MammoSight_unc_wt"
    # RUN_NAME: str = "MammoSight_unc_wt_test2"

    # ── Paths ────────────────────────────────────────────────────────────
    CSV_PATH: str = "mammo-bench_master_split.csv"
    IMAGE_ROOT: str = "/data/gaurav.bhole/FullDataset/"
    MODEL_SAVE_DIR: str = "/data/gaurav.bhole/MammoSight_unc_wt_checkpoints/"
    # MODEL_SAVE_DIR: str = "/data/gaurav.bhole/MammoSight_unc_wt_test2/"
    SAM_CHECKPOINT: Optional[str] = "/data/gaurav.bhole/pretrained_models/sam-med2d_b.pth"

    # ── Backbone ─────────────────────────────────────────────────────────
    # Choices: "efficientnet", "resnet", "vit", "swin", "biomedclip", "medimageinsight"
    BACKBONE: str = "medimageinsight"
    # Model variant passed to the loader, e.g. "efficientnet_b5", "resnet50",
    # "vit_base_patch16_224", "swin_tiny_patch4_window7_224",
    # "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    # or a local path for MedImageInsight.
    BACKBONE_NAME: str = "/data/gaurav.bhole/pretrained_models/medimageinsights/2024.09.27"

    # ── Image ────────────────────────────────────────────────────────────
    # NOTE: BiomedCLIP forces 224 regardless of this value.
    IMAGE_SIZE: int = 224

    # ── Pooling / Saliency ───────────────────────────────────────────────
    # Pooling mode for RoI-guided classification features.
    # Choices: "global"        – global average pool only (baseline)
    #          "sam_roi"       – use SAM masks for weighted pooling
    #          "learned"       – lightweight learned spatial-attention block
    POOLING_MODE: str = "sam_roi"

    # Post-hoc saliency visualisation during evaluation.
    # Choices: "none", "gradcam", "gradcam_plusplus", "score_cam"
    SALIENCY_METHOD: str = "gradcam_plusplus"

    # ── Fusion ───────────────────────────────────────────────────────────
    # Choices: "cross_attention" (proposed) | "concat" (baseline)
    FUSION_MODE: str = "cross_attention"

    # ── Loss ─────────────────────────────────────────────────────────────
    USE_DHN_NCE: bool = True
    DHN_NCE_WEIGHT: float = 0.75
    DHN_TEMPERATURE: float = 0.075
    DHN_BETA: float = 0.5
    DHN_ALPHA: float = 1.0
    DHN_PROJECTION_DIM: int = 128

    # ── Task weights ─────────────────────────────────────────────────────
    W_SEVERITY: float = 1.25
    W_DENSITY: float = 2.25
    W_BIRADS: float = 0.75
    W_ABNORMALITY: float = 0.75
    W_MOLECULAR: float = 0.5
    W_SEGMENTATION: float = 2.0

    # ── Class counts & weights ───────────────────────────────────────────
    NUM_SEVERITY_CLASSES: int = 3   # Normal / Benign / Malignant
    NUM_DENSITY_CLASSES: int = 4    # A / B / C / D
    NUM_BIRADS_CLASSES: int = 6     # 0–5
    NUM_ABNORMALITY_CLASSES: int = 3  # Normal / Mass / Calcification
    NUM_MOLECULAR_CLASSES: int = 4  # LumA / LumB / HER2 / TripleNeg

    # ── Class weights (inverse-frequency, from training distribution) ────────────
    SEVERITY_WEIGHTS    : List[float] = field(default_factory=lambda: [0.32, 1.00, 2.68])   # Normal, Benign, Malignant
    DENSITY_WEIGHTS     : List[float] = field(default_factory=lambda: [2.19, 0.65, 0.44, 2.57])  # A, B, C, D
    BIRADS_WEIGHTS      : List[float] = field(default_factory=lambda: [0.62, 0.16, 0.57, 2.67, 2.92, 4.17])  # 0,1,2,3,4,5
    ABNORMALITY_WEIGHTS : List[float] = field(default_factory=lambda: [5.99, 0.46, 1.00])   # normal, mass, calcification
    MOLECULAR_WEIGHTS   : List[float] = field(default_factory=lambda: [0.83, 0.36, 1.26, 1.33])  # LumA, LumB, HER2, TripNeg

    # ── Optimiser / Scheduler ────────────────────────────────────────────
    LR: float = 5e-6
    WEIGHT_DECAY: float = 1e-4
    BATCH_SIZE: int = 4
    GRAD_ACCUM_STEPS: int = 1
    EPOCHS: int = 50
    WARMUP_FRACTION: float = 0.05
    
    # ── Uncertainty Weighting ─────────────────────────────────────────────
    # USE_UNCERTAINTY_WEIGHTING: bool = False   # Homoscedastic uncertainty weighting (Kendall et al. 2018)
    USE_UNCERTAINTY_WEIGHTING: bool = True   # Homoscedastic uncertainty weighting (Kendall et al. 2018)

    # ── Misc ─────────────────────────────────────────────────────────────
    NUM_WORKERS: int = 4
    EVAL_INTERVAL: int = 5
    DEVICE: str = "cuda:0"
    SEED: int = 42

    # ── Helpers ──────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})