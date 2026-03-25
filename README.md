# MammoInsight

Multi-task mammography foundation model with pluggable backbone, fusion, pooling, and loss.

## Project Structure

```
MammoInsight/
├── configs/
│   └── config.py                       # All flags & hyperparameters
├── dataset/
│   └── dataset.py                       # Multi-task dataset with validity masks
├── models/
│   └── model.py                         # MammoInsightModel
├── losses/
│   ├── ordinal_loss.py                  # Weighted ordinal regression
│   └── dhn_nce_loss.py                  # DHN-NCE contrastive loss + Dice loss
├── utils/
│   ├── metrics.py                       # Classification metrics, confusion matrices
│   └── saliency.py                      # Post-hoc GradCAM / GradCAM++ / ScoreCAM
├── notebooks/
│   └── MammoInsight_explorer.ipynb       # Interactive evaluation & explainability
├── train.py                             # Training script
├── test.py                              # Evaluation script (metrics + saliency export)
├── sweep.py                             # W&B hyperparameter search
├── ARCHITECTURE_DIAGRAM_PROMPT.md       # Prompt for generating architecture figures
└── requirements.txt
```

## Quick Start

```bash
# Default training (EfficientNet-B5 + learned saliency + cross-attention + DHN-NCE)
python train.py

# Baseline (global pooling, concat fusion, no contrastive loss)
python train.py \
  --POOLING_MODE global \
  --FUSION_MODE concat \
  --USE_DHN_NCE False \
  --RUN_NAME baseline_effnet

# BiomedCLIP backbone (image size forced to 224)
python train.py \
  --BACKBONE biomedclip \
  --BACKBONE_NAME "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224" \
  --RUN_NAME biomedclip_run

# ResNet-50 with SAM-ROI pooling
python train.py \
  --BACKBONE resnet \
  --BACKBONE_NAME resnet50 \
  --POOLING_MODE sam_roi \
  --RUN_NAME resnet50_samroi

# ViT (requires timm)
python train.py \
  --BACKBONE vit \
  --BACKBONE_NAME vit_base_patch16_224 \
  --RUN_NAME vit_base

# Swin Transformer (requires timm)
python train.py \
  --BACKBONE swin \
  --BACKBONE_NAME swin_tiny_patch4_window7_224 \
  --RUN_NAME swin_tiny
```

## Hyperparameter Search

```bash
# Launch a Bayesian sweep (20 runs)
python sweep.py --count 20

# Join an existing sweep
python sweep.py --sweep_id <ID>
```

## Evaluation

```bash
# Full test-set evaluation
python test.py --checkpoint checkpoints/best_model_run.pth

# With saliency map export
python test.py --checkpoint best.pth \
  --SALIENCY_METHOD gradcam_plusplus \
  --saliency_dir ./saliency_out \
  --saliency_n 100

# Evaluate on validation set
python test.py --checkpoint best.pth --split val
```

Outputs: `test_results/predictions_test.csv`, `metrics_test.json`, and confusion matrix PNGs.

## Interactive Notebook

Open `notebooks/MammoInsight_explorer.ipynb` for:
- Architecture walkthrough with parameter breakdown
- Per-task confusion matrices and ROC curves
- Saliency map visualisation (GradCAM / GradCAM++ / ScoreCAM)
- Segmentation overlay (GT vs predicted masks)
- t-SNE / UMAP embedding visualisation
- Error analysis with calibration plots

## Feature Flags

| Flag | Options | Default |
|------|---------|---------|
| `BACKBONE` | `efficientnet`, `resnet`, `vit`, `swin`, `biomedclip`, `medimageinsight` | `efficientnet` |
| `POOLING_MODE` | `global`, `sam_roi`, `learned` | `learned` |
| `FUSION_MODE` | `cross_attention`, `concat` | `cross_attention` |
| `USE_DHN_NCE` | `True`, `False` | `True` |
| `SALIENCY_METHOD` | `none`, `gradcam`, `gradcam_plusplus`, `score_cam` | `none` |

## Post-hoc Saliency Maps

```python
from utils.saliency import compute_saliency_map

cam = compute_saliency_map(
    model, image_tensor,       # [1, 3, H, W]
    method="gradcam_plusplus",  # or "gradcam", "score_cam"
    view="cc",
)
# cam is a [H, W] numpy array in [0, 1]
```




## Detailed Data Flow with Tensor Sizes

Using `B=4` (medimageinsight batch size), `img_size=224`:
```
────────────────────────────────────────────────────────────────────────
STAGE 1 — INPUT
────────────────────────────────────────────────────────────────────────
cc_image        : [B, 3, 224, 224]   float32, ImageNet normalized
mlo_image       : [B, 3, 224, 224]   float32, ImageNet normalized
view_mask       : [B, 2]             1.0=present, 0.0=missing
cc_mask (GT)    : [B, 1, H_orig, W_orig]  binary, variable size
mlo_mask (GT)   : [B, 1, H_orig, W_orig]
has_seg         : [B]                1.0 if any GT mask exists
labels["age"]   : [B]                age/100 ∈ [0,1]

────────────────────────────────────────────────────────────────────────
STAGE 2 — BACKBONE  (forward_view, called separately for CC and MLO)
────────────────────────────────────────────────────────────────────────
MedImageInsight is a DaViT (Domain-Adaptive Vision Transformer).
It processes through alternating Conv stages and Block stages.

DaViT output (raw):  [B, N_tokens, 2048]
  where N_tokens = (224/patch_size)^2, depends on DaViT config
  For the specific model: typically [B, 196, 2048] or [B, H'*W', 2048]

────────────────────────────────────────────────────────────────────────
STAGE 3 — UNIVERSAL ADAPTER  (shared weights, applied to each view)
────────────────────────────────────────────────────────────────────────
Input raw: [B, N, 2048]
  If dim==3: reshape → [B, 2048, sqrt(N), sqrt(N)]  e.g. [B, 2048, 14, 14]

conv1 (Conv2d 2048→256, kernel=1):  [B, 256, 14, 14]
GroupNorm(32 groups, 256 ch):       [B, 256, 14, 14]
GELU activation:                    [B, 256, 14, 14]
conv2 (Conv2d 256→256, kernel=3):   [B, 256, 14, 14]
F.interpolate → (64, 64):           [B, 256, 64, 64]  ← sam_feats

global_pool = mean(sam_feats, dim=(2,3)):  [B, 256]   ← global_pool

cc_sam  = [B, 256, 64, 64]
cc_pool = [B, 256]
mlo_sam = [B, 256, 64, 64]   (same adapter, different input)
mlo_pool= [B, 256]

────────────────────────────────────────────────────────────────────────
STAGE 4 — SAM-MED2D SEGMENTATION DECODER
────────────────────────────────────────────────────────────────────────
sparse_embeddings : [B, 0, 256]    (no point/box prompts)
dense_embeddings  : [B, 256, 64, 64]  (zeros)
pos_embed         : [B, 256, 64, 64]  (zeros)

TwoWayTransformer(depth=2, embed_dim=256, mlp_dim=2048, heads=8):
  cross-attends between image tokens [B, 4096, 256] and sparse tokens

MaskDecoder output:
  mask_cc  : [B, 4, 256, 256]   (num_multimask_outputs=3, +1 = 4 masks)
  iou_preds: [B, 4]             (not used downstream)

  ↳ only mask_cc / mask_mlo are stored in `out` dict

────────────────────────────────────────────────────────────────────────
STAGE 5 — SAM ROI POOLING  (pooling_mode="sam_roi")
────────────────────────────────────────────────────────────────────────
During TRAINING (has GT masks):
  roi_cc = F.interpolate(batch["cc_mask"], (64,64), nearest):  [B, 1, 64, 64]

During INFERENCE:
  roi_cc = (sigmoid(mask_cc) > 0.5).float()
  roi_cc = F.interpolate(roi_cc, (64,64), nearest):            [B, 1, 64, 64]
  ↳ Note: mask_cc is [B,4,H,W], only first channel used implicitly via >0.5

_weighted_pool(cc_sam, roi_cc, cc_pool):
  s = roi_cc.sum(dim=(2,3)):          [B, 1]
  weighted = (cc_sam * roi_cc).sum/(s+ε):  [B, 256]
  has_roi = (s > 10).float():         [B, 1]
  pooled = has_roi*weighted + (1-has_roi)*cc_pool:  [B, 256]
  cc_feat = cat([cc_pool, pooled]):   [B, 512]

cc_feat  = [B, 512]
mlo_feat = [B, 512]

────────────────────────────────────────────────────────────────────────
STAGE 6 — CROSS-VIEW FUSION
────────────────────────────────────────────────────────────────────────
seq = stack([cc_feat, mlo_feat], dim=1):  [B, 2, 512]
pad_mask = (view_mask == 0):              [B, 2]  True=padded

MultiheadAttention(512, 8 heads, batch_first=True):
  Q=K=V=seq:  attn_out [B, 2, 512]

LayerNorm(seq + attn_out):  [B, 2, 512]
cc_out  = seq[:, 0]:  [B, 512]
mlo_out = seq[:, 1]:  [B, 512]

gate = sigmoid(Linear(1024→512)(cat([cc_out, mlo_out]))):  [B, 512]
fused = gate*cc_out + (1-gate)*mlo_out:                    [B, 512]

# Handle missing views:
has_cc  = view_mask[:, 0:1]:  [B, 1]
has_mlo = view_mask[:, 1:2]:  [B, 1]
output = where(has_cc*has_mlo > 0.5,
               fused,
               cc_out*has_cc + mlo_out*has_mlo):  [B, 512]

────────────────────────────────────────────────────────────────────────
STAGE 7 — AGE ENCODER
────────────────────────────────────────────────────────────────────────
age = labels["age"].unsqueeze(1):  [B, 1]

Linear(1→64):    [B, 64]
LayerNorm(64):   [B, 64]
GELU:            [B, 64]
Linear(64→256):  [B, 256]
LayerNorm(256):  [B, 256]

age_emb = [B, 256]

────────────────────────────────────────────────────────────────────────
STAGE 8 — COMBINED FEATURES
────────────────────────────────────────────────────────────────────────
combined = cat([fused, age_emb], dim=1):  [B, 768]
                   ↑ 512          ↑ 256

────────────────────────────────────────────────────────────────────────
STAGE 9 — TASK HEADS
────────────────────────────────────────────────────────────────────────
head_severity    : Linear(768→2)  →  [B, 2]   ordinal logits (3 classes)
head_density     : Linear(768→3)  →  [B, 3]   ordinal logits (4 classes)
head_birads      : Linear(768→5)  →  [B, 5]   ordinal logits (6 classes)
head_abnormality : Linear(768→3)  →  [B, 3]   nominal logits
head_molecular   : Linear(768→4)  →  [B, 4]   nominal logits

Ordinal decoding at inference (ordinal_logits_to_probs):
  Input [B, K-1] logits → sigmoid → cumulative probs
  → convert to [B, K] class probs → argmax → class index

────────────────────────────────────────────────────────────────────────
STAGE 10 — DHN-NCE PROJECTORS (training only, when USE_DHN_NCE=True)
────────────────────────────────────────────────────────────────────────
5 task-specific projectors (one per task), each:
  Linear(768→256):  [B, 768] → [B, 256]
  ReLU:             [B, 256]
  Linear(256→128):  [B, 128]
  LayerNorm(128):   [B, 128]
  L2-normalize:     [B, 128]  ← on unit hypersphere

Logit matrix = z @ z.T / temperature:  [B, B]
Hard-negative reweighting + log loss → scalar

────────────────────────────────────────────────────────────────────────
STAGE 11 — LOSSES (training)
────────────────────────────────────────────────────────────────────────
WeightedOrdinalRegressionLoss per sample → [B] → * validity → mean
  severity_loss, density_loss, birads_loss

CrossEntropy(reduction="none") → [B] → * validity → mean
  abnormality_loss, molecular_loss

DiceLoss + BCE:
  F.interpolate(mask_cc, orig_size) vs GT:  scalar
  same for mlo

total = 1.25*L_sev + 2.5*L_den + 0.5*L_birads
      + 0.5*L_abn + 0.5*L_mol + 2.0*L_seg
      + DHN_NCE_WEIGHT * Σ(dhn_task_losses)
