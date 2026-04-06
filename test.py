#!/usr/bin/env python3
"""
MammoFormer – Test / Evaluation Script.

Produces:
  1. Per-task metrics table (Accuracy, Balanced Accuracy, F1, MCC, AUC)
  2. Per-source-dataset breakdown
  3. Confusion matrices saved as PNGs
  4. Segmentation Dice & IoU
  5. Optional saliency map export (GradCAM / GradCAM++ / ScoreCAM)
  6. Full prediction CSV for downstream analysis

Usage
-----
  python test.py --checkpoint checkpoints/best_model_run.pth

  # With saliency maps
  python test.py --checkpoint best.pth --SALIENCY_METHOD gradcam_plusplus --saliency_dir ./saliency_out

  # Override backbone / config to match training
  python test.py --checkpoint best.pth --BACKBONE resnet --BACKBONE_NAME resnet50
"""

import os
import sys
import json
import argparse
import random
import surface_distance
from scipy.ndimage import binary_erosion
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from configs.config import Config
from dataset.dataset import MammoBenchDataset
from models.model import MammoSightModel
from losses.dhn_nce_loss import DiceLoss
from utils.metrics import (
    ordinal_logits_to_probs,
    calculate_classification_metrics,
    calculate_iou,
    plot_confusion_matrix,
    print_metrics_table,
    print_classwise_metrics
)
from utils.saliency import compute_saliency_map


# ─── CLI ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="MammoFormer Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test"],
                        help="Which split to evaluate on")
    parser.add_argument("--output_dir", type=str, default="./test_results",
                        help="Directory for all outputs")
    parser.add_argument("--saliency_dir", type=str, default=None,
                        help="Directory for saliency PNGs (None = skip)")
    parser.add_argument("--saliency_n", type=int, default=50,
                        help="Max number of saliency maps to generate")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--save_pred_masks", action="store_true",
                        help="Save predicted segmentation masks to disk")
    parser.add_argument("--mask_save_dir", type=str, default="./pred_masks",
                        help="Directory for saved predicted masks")

    # Allow all Config fields as overrides
    for name, field_obj in Config.__dataclass_fields__.items():
        ft = field_obj.type
        if ft == bool:
            parser.add_argument(
                f"--{name}",
                type=lambda x: x.lower() in ("true", "1", "yes"),
                default=None,
            )
        elif "List" in str(ft):
            parser.add_argument(f"--{name}", nargs="+", type=float, default=None)
        else:
            parser.add_argument(
                f"--{name}",
                type=type(field_obj.default) if not isinstance(field_obj.default, type) else str,
                default=None,
            )

    args = parser.parse_args()
    cfg = Config()
    for k, v in vars(args).items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)
    if cfg.BACKBONE.lower() == "biomedclip":
        cfg.IMAGE_SIZE = 224
    return args, cfg


# ─── helpers ─────────────────────────────────────────────────────────────

CLASS_NAMES = {
    "classification": ["Normal", "Benign", "Malignant"],
    "density": ["A", "B", "C", "D"],
    "birads": [str(i) for i in range(6)],
    "abnormality": ["Normal", "Mass", "Calc"],
    "molecular": ["LumA", "LumB", "HER2", "TripNeg"],
}

ORDINAL_TASKS = {"classification", "density", "birads"}
NOMINAL_TASKS = {"abnormality", "molecular"}
ALL_TASKS = list(CLASS_NAMES.keys())

def _hd95(pred: np.ndarray, gt: np.ndarray,
          spacing_mm=(1.0, 1.0)) -> float:
    assert pred.ndim == 2 and gt.ndim == 2
    pred_b = pred.astype(bool)
    gt_b   = gt.astype(bool)
    
    if not pred_b.any() or not gt_b.any():
        return np.nan
    if pred_b.all() or gt_b.all():
        return np.nan
    
    try:
        surf = surface_distance.compute_surface_distances(
            gt_b, pred_b, spacing_mm=spacing_mm
        )
        return float(surface_distance.compute_robust_hausdorff(surf, 95))
    except Exception:
        return np.nan


def _nsd(pred: np.ndarray, gt: np.ndarray,
         spacing_mm=(1.0, 1.0), tolerance_mm=5.0) -> float:
    """
    Normalized Surface Distance — fraction of surface within tolerance.
    Returns np.nan for empty masks or degenerate surface computations.
    """
    assert pred.ndim == 2 and gt.ndim == 2, \
        f"Expected 2D arrays, got pred={pred.shape}, gt={gt.shape}"
    
    pred_b = pred.astype(bool)
    gt_b   = gt.astype(bool)
    
    # Both masks must have foreground pixels AND not be fully filled
    # (fully filled mask has no surface)
    if not pred_b.any() or not gt_b.any():
        return np.nan
    if pred_b.all() or gt_b.all():
        return np.nan
    
    try:
        surf = surface_distance.compute_surface_distances(
            gt_b, pred_b, spacing_mm=spacing_mm
        )
        result = surface_distance.compute_surface_overlap_at_tolerance(
            surf, tolerance_mm
        )
        # result is (overlap_gt, overlap_pred) — both floats
        if not isinstance(result, (tuple, list)) or len(result) < 2:
            return np.nan
        gt_overlap, pred_overlap = result[0], result[1]
        return float((gt_overlap + pred_overlap) / 2.0)
    except Exception:
        return np.nan


def _to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
        elif isinstance(v, dict):
            for sk, sv in v.items():
                if isinstance(sv, torch.Tensor):
                    batch[k][sk] = sv.to(device)


# ─── main evaluation ────────────────────────────────────────────────────

def main():
    args, cfg = parse_args()

    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.save_pred_masks:
        os.makedirs(args.mask_save_dir, exist_ok=True)

    # ── dataset ──────────────────────────────────────────────────────────
    ds = MammoBenchDataset(
        cfg.CSV_PATH, args.split, cfg.IMAGE_ROOT,
        img_size=cfg.IMAGE_SIZE, backbone_type=cfg.BACKBONE,
    )
    bs = args.batch_size or cfg.BATCH_SIZE
    loader = DataLoader(ds, bs, shuffle=False, num_workers=cfg.NUM_WORKERS)
    print(f"\nEvaluating on {args.split} split: {len(ds)} samples\n")

    # ── model ────────────────────────────────────────────────────────────
    model = MammoSightModel(
        backbone_type=cfg.BACKBONE,
        backbone_name=cfg.BACKBONE_NAME,
        sam_checkpoint_path=cfg.SAM_CHECKPOINT,
        pooling_mode=cfg.POOLING_MODE,
        fusion_mode=cfg.FUSION_MODE,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}\n")

    # ── collectors ───────────────────────────────────────────────────────
    global_coll = {t: {"preds": [], "targs": [], "probs": []} for t in ALL_TASKS}
    per_ds_coll = defaultdict(lambda: {t: {"preds": [], "targs": [], "probs": []} for t in ALL_TASKS})

    # seg_dice, seg_iou = [], []
    seg_dice, seg_iou, seg_hd95, seg_nsd = [], [], [], []
    dice_fn = DiceLoss()

    # Per-sample records for CSV export
    records = []

    # ── inference ────────────────────────────────────────────────────────
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            _to_device(batch, device)
            out = model(batch)
            B = out["classification"].shape[0]

            # Decode predictions
            preds_dict = {}
            probs_dict = {}
            for task in ORDINAL_TASKS:
                probs = ordinal_logits_to_probs(out[task])
                preds_dict[task] = probs.argmax(1).cpu().numpy()
                probs_dict[task] = probs.cpu().numpy()
            for task in NOMINAL_TASKS:
                probs = F.softmax(out[task], dim=1)
                preds_dict[task] = probs.argmax(1).cpu().numpy()
                probs_dict[task] = probs.cpu().numpy()

            # Collect per-sample
            for i in range(B):
                bid = batch["breast_id"][i]
                src = batch["source_dataset"][i]
                rec = {"breast_id": bid, "source_dataset": src}

                for task in ALL_TASKS:
                    targ = int(batch["labels"][task][i].cpu().item())
                    valid = float(batch["validity"][task][i].cpu().item())
                    pred = int(preds_dict[task][i])
                    prob = probs_dict[task][i].tolist()

                    rec[f"{task}_target"] = targ
                    rec[f"{task}_pred"] = pred
                    rec[f"{task}_valid"] = valid
                    rec[f"{task}_probs"] = prob

                    if valid > 0:
                        for coll in (global_coll, per_ds_coll[src]):
                            coll[task]["preds"].append(pred)
                            coll[task]["targs"].append(targ)
                            coll[task]["probs"].append(probs_dict[task][i])

                records.append(rec)

            # Segmentation
            if batch["has_seg"].sum() > 0 and out["pred_mask_cc"] is not None:
                tsz = batch["cc_mask"].shape[-2:]
                for view in ("cc", "mlo"):
                    pk = f"pred_mask_{view}"
                    if out.get(pk) is not None:
                        pred_m = F.interpolate(out[pk], tsz, mode="bilinear",
                                            align_corners=False)
                        gt_m = batch[f"{view}_mask"]
                        if gt_m.sum() > 0:
                            seg_dice.append(1.0 - dice_fn(pred_m, gt_m).item())
                            seg_iou.append(calculate_iou(pred_m, gt_m))

                            # squeeze channel dim only — keeps batch dim intact
                            pred_np = (torch.sigmoid(pred_m) > 0.5).squeeze(1).cpu().numpy()  # (B, H, W)
                            gt_np   = (gt_m > 0.5).squeeze(1).cpu().numpy()                   # (B, H, W)

                            for b in range(pred_np.shape[0]):
                                seg_hd95.append(_hd95(pred_np[b], gt_np[b]))
                                seg_nsd.append(_nsd(pred_np[b], gt_np[b], tolerance_mm=5.0))

                                if args.save_pred_masks:
                                    src_b    = batch["source_dataset"][b]
                                    bid_b    = batch["breast_id"][b]
                                    safe_src = str(src_b).replace("/", "_")
                                    safe_bid = str(bid_b).replace("/", "_")
                                    fname    = f"{safe_src}_{safe_bid}_{view}_ROI.png"
                                    save_p   = os.path.join(args.mask_save_dir, fname)
                                    pred_uint8 = (pred_np[b] * 255).astype(np.uint8)
                                    from PIL import Image as PILImage
                                    PILImage.fromarray(pred_uint8, mode="L").save(save_p)

    # ── global metrics ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"GLOBAL METRICS  ({args.split} split, {len(ds)} samples)")
    print("=" * 80)

    all_metrics = {}
    for task in ALL_TASKS:
        d = global_coll[task]
        if not d["targs"]:
            continue
        t = np.array(d["targs"])
        p = np.array(d["preds"])
        pb = np.vstack(d["probs"])
        m, report = calculate_classification_metrics(t, p, pb, task, CLASS_NAMES[task], phase="test")
        all_metrics.update(m)
        
        print_classwise_metrics(report, task, CLASS_NAMES[task]) 

        # Save confusion matrix
        fig = plot_confusion_matrix(t, p, CLASS_NAMES[task], task)
        fig.savefig(os.path.join(args.output_dir, f"cm_{task}.png"), dpi=150)
        plt.close(fig)

    print_metrics_table(all_metrics, ALL_TASKS, phase="test")
    if seg_dice:
        d_mean  = float(np.mean(seg_dice))
        i_mean  = float(np.mean(seg_iou))
        h_mean  = float(np.nanmean(seg_hd95)) if seg_hd95 else float('nan')
        n_mean  = float(np.nanmean(seg_nsd))  if seg_nsd  else float('nan')
        all_metrics["test/seg_dice"] = d_mean
        all_metrics["test/seg_iou"]  = i_mean
        all_metrics["test/seg_hd95"] = h_mean
        all_metrics["test/seg_nsd"]  = n_mean
        print(
            f"\n{'SEGMENTATION':<15} | "
            f"Dice: {d_mean:.4f} | IoU: {i_mean:.4f} | "
            f"HD95: {h_mean:.2f}px | NSD@5mm: {n_mean:.4f}"
        )

    # ── per-dataset breakdown ────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PER-DATASET BREAKDOWN (Severity F1 Macro)")
    print("=" * 80)
    print(f"{'Dataset':<25} | {'N':<8} | {'Acc':<8} | {'F1 Mac':<8} | {'MCC':<8} | {'AUC':<8} | {'WCK':<8}")
    print("-" * 75)

    ds_summary = {}
    for src, coll in sorted(per_ds_coll.items()):
        d = coll["classification"]
        if not d["targs"]:
            continue
        t = np.array(d["targs"])
        p = np.array(d["preds"])
        pb = np.vstack(d["probs"])
        m, _ = calculate_classification_metrics(t, p, pb, "classification", CLASS_NAMES["classification"], phase="test")

        acc = m.get("test/classification_acc", 0)
        f1m = m.get("test/classification_f1_macro", 0)
        mcc = m.get("test/classification_mcc", 0)
        auc = m.get("test/classification_auc", 0)
        wck = m.get("test/classification_wck",      None)

        print(f"{src:<25} | {len(t):<8} | {acc:<8.4f} | {f1m:<8.4f} | {mcc:<8.4f} | {auc:<8.4f} | {wck:<8.4f}" if wck is not None else f"{src:<25} | {len(t):<8} | {acc:<8.4f} | {f1m:<8.4f} | {mcc:<8.4f} | {auc:<8.4f}")
        ds_summary[src] = {"n": len(t), "acc": acc, "f1_macro": f1m, "mcc": mcc, "auc": auc, "wck": wck}

    # ── save CSV of all predictions ──────────────────────────────────────
    csv_path = os.path.join(args.output_dir, f"predictions_{args.split}.csv")
    # Flatten probs for CSV
    flat_records = []
    for r in records:
        flat = {k: v for k, v in r.items() if not k.endswith("_probs")}
        for task in ALL_TASKS:
            probs = r[f"{task}_probs"]
            for ci, p in enumerate(probs):
                flat[f"{task}_prob_class{ci}"] = round(p, 5)
        flat_records.append(flat)
    pd.DataFrame(flat_records).to_csv(csv_path, index=False)
    print(f"\nPredictions saved → {csv_path}")

    # ── save metrics JSON ────────────────────────────────────────────────
    json_path = os.path.join(args.output_dir, f"metrics_{args.split}.json")
    out_json = {
        "global": {k: float(v) for k, v in all_metrics.items()},
        "per_dataset": ds_summary,
        "segmentation": {
            "dice":      float(np.mean(seg_dice))        if seg_dice else None,
            "iou":       float(np.mean(seg_iou))         if seg_iou  else None,
            "hd95_mean": float(np.nanmean(seg_hd95))     if seg_hd95 else None,
            "hd95_median": float(np.nanmedian(seg_hd95)) if seg_hd95 else None,
            "nsd_mean":  float(np.nanmean(seg_nsd))      if seg_nsd  else None,
            "n_samples": len(seg_dice),
        },
        "config": cfg.to_dict(),
    }
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"Metrics saved   → {json_path}")

    # ── saliency maps ────────────────────────────────────────────────────
    if args.saliency_dir and cfg.SALIENCY_METHOD != "none":
        _generate_saliency(
            model, ds, device, cfg,
            args.saliency_dir, args.saliency_n,
        )

    print("\n✓ Evaluation complete.\n")


# ─── saliency export ────────────────────────────────────────────────────

def _generate_saliency(model, dataset, device, cfg, out_dir, n_samples):
    """Generate and save saliency map overlays for n_samples."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nGenerating {cfg.SALIENCY_METHOD} saliency maps for {n_samples} samples...")

    # We need the model in eval mode but with gradients enabled for GradCAM
    model.eval()

    indices = list(range(min(n_samples, len(dataset))))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for idx in tqdm(indices, desc="Saliency"):
        sample = dataset[idx]
        bid = sample["breast_id"]

        for view in ("cc", "mlo"):
            img_key = f"{view}_image"
            img_t = sample[img_key].unsqueeze(0).to(device)

            # Skip empty views
            view_idx = 0 if view == "cc" else 1
            if sample["view_mask"][view_idx] < 0.5:
                continue

            cam = compute_saliency_map(
                model, img_t,
                method=cfg.SALIENCY_METHOD,
                target_class=None,
                view=view,
            )
            if cam is None:
                continue

            # Denormalise image for overlay
            raw = img_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            raw = (raw * std + mean).clip(0, 1)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(raw)
            axes[0].set_title(f"{view.upper()} Original")
            axes[0].axis("off")

            axes[1].imshow(cam, cmap="jet")
            axes[1].set_title(f"{cfg.SALIENCY_METHOD}")
            axes[1].axis("off")

            axes[2].imshow(raw)
            axes[2].imshow(cam, cmap="jet", alpha=0.4)
            axes[2].set_title("Overlay")
            axes[2].axis("off")

            safe_bid = bid.replace("/", "_").replace("\\", "_")
            fig.savefig(
                os.path.join(out_dir, f"{safe_bid}_{view}.png"),
                dpi=120, bbox_inches="tight",
            )
            plt.close(fig)

    print(f"Saliency maps saved → {out_dir}/")


if __name__ == "__main__":
    main()