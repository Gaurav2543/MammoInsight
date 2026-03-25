#!/usr/bin/env python3
"""
MammoFormer – Training Script.

Usage
-----
  # Default (uses configs/config.py)
  python train.py

  # Override via CLI
  python train.py --BACKBONE resnet --BACKBONE_NAME resnet50 --FUSION_MODE concat

  # Minimal baseline (no DHN-NCE, global pooling, concat fusion)
  python train.py --USE_DHN_NCE False --POOLING_MODE global --FUSION_MODE concat
"""

import os
import sys
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import wandb

from sklearn.metrics import f1_score as _f1
from configs.config import Config
from dataset.dataset import MammoBenchDataset
from models.model import MammoSightModel
from losses.uncertainty_loss import MultiTaskUncertaintyLoss
from losses.ordinal_loss import WeightedOrdinalRegressionLoss
from losses.dhn_nce_loss import DHN_NCE_Loss, DiceLoss
from utils.metrics import (
    ordinal_logits_to_probs,
    calculate_classification_metrics,
    calculate_iou,
    plot_confusion_matrix,
    print_metrics_table,
)


# ── helpers ──────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> Config:
    """Parse CLI overrides and return a populated Config."""
    parser = argparse.ArgumentParser(description="MammoFormer Training")
    for name, field_obj in Config.__dataclass_fields__.items():
        ft = field_obj.type
        if ft == bool:
            parser.add_argument(f"--{name}", type=lambda x: x.lower() in ("true", "1", "yes"),
                                default=None)
        elif "List" in str(ft):
            parser.add_argument(f"--{name}", nargs="+", type=float, default=None)
        else:
            parser.add_argument(f"--{name}", type=type(field_obj.default)
                                if not isinstance(field_obj.default, type) else str,
                                default=None)
    args = parser.parse_args()
    cfg = Config()
    for k, v in vars(args).items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg


def validate_labels(dataset, split_name: str):
    """Quick sanity check on the first 100 samples."""
    limits = {"classification": 2, "density": 3, "birads": 5,
              "abnormality": 2, "molecular": 3}
    for i in range(min(100, len(dataset))):
        s = dataset[i]
        for task, mx in limits.items():
            if s["validity"][task] > 0 and (s["labels"][task] < 0 or s["labels"][task] > mx):
                raise ValueError(
                    f"{split_name} sample {i}: {task} label {s['labels'][task]} "
                    f"out of range [0, {mx}]"
                )
    print(f"  ✓ {split_name} labels validated")


# ── training ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, loss_fns, dhn_nce,
                    scaler, device, epoch, scheduler, cfg,
                    full_eval=False, unc_loss=None):
    model.train()
    running, n_steps = 0.0, 0

    collectors = None
    if full_eval:
        collectors = {t: {"preds": [], "targs": [], "probs": []}
                      for t in ("classification", "density", "birads",
                                "abnormality", "molecular")}

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    optimizer.zero_grad()

    for step, batch in enumerate(pbar):
        _to_device(batch, device)

        with torch.cuda.amp.autocast():
            out = model(batch, return_features=cfg.USE_DHN_NCE)

            # ── ordinal losses ───────────────────────────────────────────
            loss_s = _ordinal_loss(loss_fns["ord_severity"], out["classification"],
                                   batch, "classification", dhn_nce, cfg, "severity", out)
            loss_d = _ordinal_loss(loss_fns["ord_density"], out["density"],
                                   batch, "density", dhn_nce, cfg, "density", out)
            loss_b = _ordinal_loss(loss_fns["ord_birads"], out["birads"],
                                   batch, "birads", dhn_nce, cfg, "birads", out)

            # ── nominal losses ───────────────────────────────────────────
            loss_a = _nominal_loss(out["abnormality"], batch, "abnormality",
                        dhn_nce, cfg, out, ce_fn=loss_fns["ce_abnormality"])
            loss_m = _nominal_loss(out["molecular"], batch, "molecular",
                        dhn_nce, cfg, out, ce_fn=loss_fns["ce_molecular"])

            # ── segmentation ─────────────────────────────────────────────
            loss_seg = _seg_loss(out, batch, loss_fns["dice"], device)
            
            if unc_loss is not None:
                per_task = {
                    "severity":     loss_s,
                    "density":      loss_d,
                    "birads":       loss_b,
                    "abnormality":  loss_a,
                    "molecular":    loss_m,
                    "segmentation": loss_seg,
                }
                total, eff_weights = unc_loss(per_task)
                # Log learned weights every 50 steps so you can watch them evolve
                if step % 50 == 0:
                    import wandb
                    wandb.log({"train/unc_" + k: v for k, v in eff_weights.items()})
            else:
                total = (cfg.W_SEVERITY * loss_s + cfg.W_DENSITY * loss_d +
                        cfg.W_BIRADS * loss_b + cfg.W_ABNORMALITY * loss_a +
                        cfg.W_MOLECULAR * loss_m + cfg.W_SEGMENTATION * loss_seg)

            total = total / cfg.GRAD_ACCUM_STEPS

        scaler.scale(total).backward()

        if (step + 1) % cfg.GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

        running += total.item() * cfg.GRAD_ACCUM_STEPS
        n_steps += 1
        pbar.set_postfix(loss=f"{total.item() * cfg.GRAD_ACCUM_STEPS:.4f}")

        if full_eval:
            _collect(out, batch, collectors)

    avg = running / max(n_steps, 1)

    if full_eval and collectors:
        _log_full_metrics(collectors, epoch, phase="train")

    return avg

def _safe_concat(lst):
    """Concatenate list of arrays, return None if result is empty."""
    if not lst:
        return None
    arr = np.concatenate(lst)
    return arr if len(arr) > 0 else None

# ── validation ───────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, device, epoch, full_eval, cfg):
    model.eval()

    collectors = {t: {"preds": [], "targs": [], "probs": []}
                  for t in ("classification", "density", "birads",
                            "abnormality", "molecular")}
    seg_dice, seg_iou = [], []
    dice_fn = DiceLoss()
    running_loss, n_batches = 0.0, 0

    severity_w = torch.tensor(cfg.SEVERITY_WEIGHTS, device=device)
    ord_sev = WeightedOrdinalRegressionLoss(cfg.NUM_SEVERITY_CLASSES, severity_w).to(device)

    for batch in tqdm(loader, desc=f"Epoch {epoch} [Val]"):
        _to_device(batch, device)
        out = model(batch)

        # lightweight val loss (severity only for logging)
        ls = ord_sev(out["classification"], batch["labels"]["classification"].long())
        ls = (ls * batch["validity"]["classification"]).mean()
        running_loss += ls.item()
        n_batches += 1

        _collect(out, batch, collectors)

        # segmentation
        if batch["has_seg"].sum() > 0 and out["pred_mask_cc"] is not None:
            tsz = batch["cc_mask"].shape[-2:]
            for view in ("cc", "mlo"):
                pk = f"pred_mask_{view}"
                if out.get(pk) is not None:
                    pred = F.interpolate(out[pk], tsz, mode="bilinear", align_corners=False)
                    gt = batch[f"{view}_mask"]
                    if gt.sum() > 0:
                        seg_dice.append(1.0 - dice_fn(pred, gt).item())
                        seg_iou.append(calculate_iou(pred, gt))

    task_weights = {
        "classification": 0.25,
        "density": 0.275,
        "birads": 0.175,
        "abnormality": 0.20,
        "molecular": 0.10,
    }
    
    prim = 0.0
    for task, w in task_weights.items():
        t = _safe_concat(collectors[task]["targs"])
        p = _safe_concat(collectors[task]["preds"])
        if t is not None and p is not None:
            prim += w * _f1(t, p, average="macro", zero_division=0)

    log = {"val/epoch_loss": running_loss / max(n_batches, 1), "epoch": epoch,
        "val/composite_score": prim}

    for task in task_weights:
        t = _safe_concat(collectors[task]["targs"])
        p = _safe_concat(collectors[task]["preds"])
        if t is not None and p is not None:
            log[f"val/{task}_f1_for_save"] = _f1(t, p, average="macro", zero_division=0)

    # Include seg dice
    if seg_dice:
        prim += 0.0 * np.mean(seg_dice)  # adjust weight if desired

    if full_eval:
        extra = _log_full_metrics(collectors, epoch, phase="val", seg_dice=seg_dice, seg_iou=seg_iou)
        log.update(extra)

    wandb.log(log)
    return prim


# ── loss helpers ─────────────────────────────────────────────────────────

def _to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
        elif isinstance(v, dict):
            for sk, sv in v.items():
                if isinstance(sv, torch.Tensor):
                    batch[k][sk] = sv.to(device)


def _ordinal_loss(loss_fn, logits, batch, task, dhn_nce, cfg, dhn_task, out=None):
    loss = loss_fn(logits, batch["labels"][task].long())
    if cfg.USE_DHN_NCE and dhn_nce is not None and out is not None:
        feats = out.get("combined_features", logits)
        dhn = dhn_nce(feats, batch["labels"][task],
                      batch["validity"][task], dhn_task)
        loss = loss + cfg.DHN_NCE_WEIGHT * dhn
    return (loss * batch["validity"][task]).mean()

def _nominal_loss(logits, batch, task, dhn_nce, cfg, out=None, ce_fn=None):
    if ce_fn is not None:
        ce = ce_fn(logits, batch["labels"][task].long())
    else:
        ce = F.cross_entropy(logits, batch["labels"][task].long(), reduction="none")
    if cfg.USE_DHN_NCE and dhn_nce is not None and out is not None:
        dhn_task = "abnormality" if task == "abnormality" else "molecular"
        feats = out.get("combined_features", logits)
        dhn = dhn_nce(feats, batch["labels"][task],
                      batch["validity"][task], dhn_task)
        ce = ce + cfg.DHN_NCE_WEIGHT * dhn
    return (ce * batch["validity"][task]).mean()

def _seg_loss(out, batch, dice_fn, device):
    loss = torch.tensor(0.0, device=device)
    if batch["has_seg"].sum() == 0 or out["pred_mask_cc"] is None:
        return loss
    tsz = batch["cc_mask"].shape[-2:]
    for view in ("cc", "mlo"):
        pk = f"pred_mask_{view}"
        if out.get(pk) is not None:
            pred = F.interpolate(out[pk], tsz, mode="bilinear", align_corners=False)
            gt = batch[f"{view}_mask"]
            loss += dice_fn(pred, gt) + F.binary_cross_entropy_with_logits(pred, gt)
    return loss


# ── metric collection ────────────────────────────────────────────────────

def _collect(out, batch, collectors):
    with torch.no_grad():
        for key in ("classification", "density", "birads"):
            probs = ordinal_logits_to_probs(out[key])
            preds = probs.argmax(1).cpu().numpy()
            targs = batch["labels"][key].cpu().numpy()
            valid = batch["validity"][key].bool().cpu().numpy()
            collectors[key]["preds"].append(preds[valid])
            collectors[key]["targs"].append(targs[valid])
            collectors[key]["probs"].append(probs[valid].cpu().numpy())

        for key in ("abnormality", "molecular"):
            probs = F.softmax(out[key], dim=1)
            preds = probs.argmax(1).cpu().numpy()
            targs = batch["labels"][key].cpu().numpy()
            valid = batch["validity"][key].bool().cpu().numpy()
            collectors[key]["preds"].append(preds[valid])
            collectors[key]["targs"].append(targs[valid])
            collectors[key]["probs"].append(probs[valid].cpu().numpy())


CLASS_NAMES = {
    "classification": ["Normal", "Benign", "Malignant"],
    "density": ["A", "B", "C", "D"],
    "birads": [str(i) for i in range(6)],
    "abnormality": ["Normal", "Mass", "Calc"],
    "molecular": ["LumA", "LumB", "HER2", "TripNeg"],
}

def _log_full_metrics(collectors, epoch, phase, seg_dice=None, seg_iou=None):
    """Compute and log detailed per-task metrics; return the log dict."""
    log_dict = {}
    print(f"\n{'=' * 20} EPOCH {epoch} {phase.upper()} METRICS {'=' * 20}")

    for task, data in collectors.items():
        t = _safe_concat(data["targs"])
        p = _safe_concat(data["preds"])
        pb_list = [x for x in data["probs"] if len(x) > 0]
        
        if t is None or p is None or not pb_list:
            continue
            
        pb = np.vstack(pb_list)
        metrics, _ = calculate_classification_metrics(
            t, p, pb, task, CLASS_NAMES[task], phase=phase,
        )
        log_dict.update(metrics)
        fig = plot_confusion_matrix(t, p, CLASS_NAMES[task], task)
        log_dict[f"{phase}/{task}_cm"] = wandb.Image(fig)

    if seg_dice:
        d, i = np.mean(seg_dice), np.mean(seg_iou)
        log_dict[f"{phase}/seg_dice"] = d
        log_dict[f"{phase}/seg_iou"] = i
        print(f"{'SEGMENTATION':<15} | Dice: {d:.4f} | IoU: {i:.4f}")

    print_metrics_table(log_dict, list(CLASS_NAMES.keys()), phase=phase)
    print(f"{'=' * 70}\n")
    return log_dict


# ── main ─────────────────────────────────────────────────────────────────

def main():
    cfg = parse_args()

    # Force 224 for BiomedCLIP
    if cfg.BACKBONE.lower() == "biomedclip":
        cfg.IMAGE_SIZE = 224

    set_seed(cfg.SEED)

    wandb.init(project=cfg.PROJECT_NAME, name=cfg.RUN_NAME, config=cfg.to_dict())
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.MODEL_SAVE_DIR, exist_ok=True)

    # Print config
    print("\n" + "=" * 70 + "\nCONFIGURATION\n" + "=" * 70)
    for k, v in cfg.to_dict().items():
        print(f"  {k:<25}: {v}")
    print("=" * 70 + "\n")

    # ── data ─────────────────────────────────────────────────────────────
    train_ds = MammoBenchDataset(
        cfg.CSV_PATH, "train", cfg.IMAGE_ROOT, img_size=cfg.IMAGE_SIZE,
        backbone_type=cfg.BACKBONE,
    )
    val_ds = MammoBenchDataset(
        cfg.CSV_PATH, "val", cfg.IMAGE_ROOT, img_size=cfg.IMAGE_SIZE,
        backbone_type=cfg.BACKBONE,
    )
    validate_labels(train_ds, "train")
    validate_labels(val_ds, "val")

    train_loader = DataLoader(train_ds, cfg.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, cfg.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.NUM_WORKERS)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── model ────────────────────────────────────────────────────────────
    model = MammoSightModel(
        backbone_type=cfg.BACKBONE,
        backbone_name=cfg.BACKBONE_NAME,
        sam_checkpoint_path=cfg.SAM_CHECKPOINT,
        pooling_mode=cfg.POOLING_MODE,
        fusion_mode=cfg.FUSION_MODE,
    ).to(device)
    
    # ── losses ───────────────────────────────────────────────────────────
    sev_w = torch.tensor(cfg.SEVERITY_WEIGHTS,    device=device)
    den_w = torch.tensor(cfg.DENSITY_WEIGHTS,     device=device)
    bir_w = torch.tensor(cfg.BIRADS_WEIGHTS,      device=device)
    abn_w = torch.tensor(cfg.ABNORMALITY_WEIGHTS, device=device)
    mol_w = torch.tensor(cfg.MOLECULAR_WEIGHTS,   device=device)

    loss_fns = {
        "ord_severity":   WeightedOrdinalRegressionLoss(cfg.NUM_SEVERITY_CLASSES, sev_w).to(device),
        "ord_density":    WeightedOrdinalRegressionLoss(cfg.NUM_DENSITY_CLASSES,  den_w).to(device),
        "ord_birads":     WeightedOrdinalRegressionLoss(cfg.NUM_BIRADS_CLASSES,   bir_w).to(device),
        "ce_abnormality": nn.CrossEntropyLoss(weight=abn_w, reduction="none"),
        "ce_molecular":   nn.CrossEntropyLoss(weight=mol_w, reduction="none"),
        "dice":           DiceLoss().to(device),
    }

    # ── unc_loss (instantiate before optimizer) ───────────────────────────
    task_types = {
        "severity":     "regression",
        "density":      "regression",
        "birads":       "regression",
        "abnormality":  "classification",
        "molecular":    "classification",
        "segmentation": "regression",
    }

    task_types = {
        "severity":     "regression",    # ordinal
        "density":      "regression",    # ordinal
        "birads":       "regression",    # ordinal
        "abnormality":  "classification",
        "molecular":    "classification",
        "segmentation": "regression",    # dice is regression-like
    }

    unc_loss = None
    if cfg.USE_UNCERTAINTY_WEIGHTING:
        unc_loss = MultiTaskUncertaintyLoss(
            task_names=list(task_types.keys()),
            task_types=task_types,
        ).to(device)
        # Add log_vars to optimizer so they're learned
        optimizer = optim.AdamW(
            [
                {"params": unc_loss.parameters(),   "lr": 1e-3},  # higher LR for task weights
            ],
            weight_decay=cfg.WEIGHT_DECAY,
        )
        print("Uncertainty weighting enabled (Kendall et al. 2018)")

    # ── dhn_nce (instantiate before optimizer) ───────────────────────────
    dhn_nce = None
    if cfg.USE_DHN_NCE:
        feat_dim = model.head_severity.in_features
        dhn_nce = DHN_NCE_Loss(
            temperature=cfg.DHN_TEMPERATURE, beta=cfg.DHN_BETA,
            alpha=cfg.DHN_ALPHA, projection_dim=cfg.DHN_PROJECTION_DIM,
            feature_dim=feat_dim,
        ).to(device)
        print(f"DHN-NCE enabled  (weight={cfg.DHN_NCE_WEIGHT}, feat_dim={feat_dim})")

    # ── optimizer (build after all modules exist) ─────────────────────────
    # param_groups = [{"params": model.parameters(), "lr": cfg.LR}]
    # if unc_loss is not None:
    #     param_groups.append({"params": unc_loss.parameters(), "lr": 1e-3})
    # if dhn_nce is not None:
    #     param_groups.append({"params": dhn_nce.parameters(), "lr": cfg.LR})

    optimizer = optim.AdamW(params=model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    # ── scaler & scheduler ────────────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler()

    steps_per_epoch = len(train_loader) // cfg.GRAD_ACCUM_STEPS
    total_steps = cfg.EPOCHS * steps_per_epoch
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * cfg.WARMUP_FRACTION),
        num_training_steps=total_steps,
    )

    # # ── losses ───────────────────────────────────────────────────────────
    # sev_w = torch.tensor(cfg.SEVERITY_WEIGHTS,    device=device)
    # den_w = torch.tensor(cfg.DENSITY_WEIGHTS,     device=device)
    # bir_w = torch.tensor(cfg.BIRADS_WEIGHTS,      device=device)
    # abn_w = torch.tensor(cfg.ABNORMALITY_WEIGHTS, device=device)
    # mol_w = torch.tensor(cfg.MOLECULAR_WEIGHTS,   device=device)

    # loss_fns = {
    #     "ord_severity":  WeightedOrdinalRegressionLoss(cfg.NUM_SEVERITY_CLASSES, sev_w).to(device),
    #     "ord_density":   WeightedOrdinalRegressionLoss(cfg.NUM_DENSITY_CLASSES,  den_w).to(device),
    #     "ord_birads":    WeightedOrdinalRegressionLoss(cfg.NUM_BIRADS_CLASSES,   bir_w).to(device),
    #     "ce_abnormality": nn.CrossEntropyLoss(weight=abn_w, reduction="none"),
    #     "ce_molecular":   nn.CrossEntropyLoss(weight=mol_w, reduction="none"),
    #     "dice":          DiceLoss().to(device),
    # }
    
    # # ── CHANGE 2: in main(), after loss_fns dict, instantiate unc_loss ───
    # task_types = {
    #     "severity":     "regression",    # ordinal
    #     "density":      "regression",    # ordinal
    #     "birads":       "regression",    # ordinal
    #     "abnormality":  "classification",
    #     "molecular":    "classification",
    #     "segmentation": "regression",    # dice is regression-like
    # }

    # unc_loss = None
    # if cfg.USE_UNCERTAINTY_WEIGHTING:
    #     unc_loss = MultiTaskUncertaintyLoss(
    #         task_names=list(task_types.keys()),
    #         task_types=task_types,
    #     ).to(device)
    #     # Add log_vars to optimizer so they're learned
    #     optimizer = optim.AdamW(
    #         [
    #             {"params": model.parameters(),      "lr": cfg.LR},
    #             {"params": unc_loss.parameters(),   "lr": 1e-4},  # higher LR for task weights
    #         ],
    #         weight_decay=cfg.WEIGHT_DECAY,
    #     )
    #     print("Uncertainty weighting enabled (Kendall et al. 2018)")

    # dhn_nce = None
    # if cfg.USE_DHN_NCE:
    #     # determine feature dim from model
    #     feat_dim = model.head_severity.in_features
    #     dhn_nce = DHN_NCE_Loss(
    #         temperature=cfg.DHN_TEMPERATURE, beta=cfg.DHN_BETA,
    #         alpha=cfg.DHN_ALPHA, projection_dim=cfg.DHN_PROJECTION_DIM,
    #         feature_dim=feat_dim,
    #     ).to(device)
    #     print(f"DHN-NCE enabled  (weight={cfg.DHN_NCE_WEIGHT}, feat_dim={feat_dim})")
        
    # # ── optimiser & scheduler ────────────────────────────────────────────
    # if not cfg.USE_UNCERTAINTY_WEIGHTING:
    #     optimizer = optim.AdamW(model.parameters(), lr=cfg.LR,
    #                         weight_decay=cfg.WEIGHT_DECAY)
    # scaler = torch.cuda.amp.GradScaler()

    # steps_per_epoch = len(train_loader) // cfg.GRAD_ACCUM_STEPS
    # total_steps = cfg.EPOCHS * steps_per_epoch
    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=int(total_steps * cfg.WARMUP_FRACTION),
    #     num_training_steps=total_steps,
    # )

    # ── training loop ────────────────────────────────────────────────────
    best_f1 = 0.0
    for epoch in range(cfg.EPOCHS):
        do_full = (epoch + 1) % cfg.EVAL_INTERVAL == 0

        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fns, dhn_nce,
            scaler, device, epoch, scheduler, cfg, full_eval=do_full, unc_loss=unc_loss,
        )

        val_f1 = validate(model, val_loader, device, epoch,
                          full_eval=do_full, cfg=cfg)
        
        lr_now = scheduler.get_last_lr()[0]
        wandb.log({
            "train/epoch_loss": train_loss,
            "train/lr": lr_now,
            "val/composite_score": val_f1,
            "epoch": epoch,
        })

        print(f"\nEpoch {epoch}  |  Train Loss: {train_loss:.4f}  |  "
            f"Val Composite: {val_f1:.4f}  |  LR: {lr_now:.2e}\n")

        if val_f1 > best_f1:
            best_f1 = val_f1
            path = os.path.join(cfg.MODEL_SAVE_DIR,
                                f"best_model_{cfg.RUN_NAME}.pth")
            torch.save(model.state_dict(), path)
            print(f"  ✓ Saved best model (F1={best_f1:.4f})")
        
        # At the end of each epoch in main():
        epoch_path = os.path.join(cfg.MODEL_SAVE_DIR, f"checkpoint_ep{epoch}.pth")
        torch.save(model.state_dict(), epoch_path)
        # Keep only last 3
        for old_ep in range(epoch - 3):
            old_path = os.path.join(cfg.MODEL_SAVE_DIR, f"checkpoint_ep{old_ep}.pth")
            if os.path.exists(old_path):
                os.remove(old_path)

    wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    main()