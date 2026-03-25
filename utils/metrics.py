"""
Evaluation metrics for multi-task classification and segmentation.
"""

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    cohen_kappa_score
)

ORDINAL_TASKS = {"classification", "density", "birads"} 

# ── ordinal decoding ────────────────────────────────────────────────────

def ordinal_logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """Convert ``[B, K-1]`` ordinal logits to ``[B, K]`` class probabilities."""
    sig = torch.sigmoid(logits)
    B, Km1 = sig.shape
    K = Km1 + 1
    probs = torch.zeros(B, K, device=logits.device)
    probs[:, 0] = 1.0 - sig[:, 0]
    for k in range(1, K - 1):
        probs[:, k] = sig[:, k - 1] - sig[:, k]
    probs[:, -1] = sig[:, -1]
    return probs


# ── segmentation ─────────────────────────────────────────────────────────

def calculate_iou(preds, targets, smooth=1e-6):
    """Mean IoU from logits and binary masks."""
    pred_mask = (torch.sigmoid(preds) > 0.5).float()
    inter = (pred_mask * targets).sum(dim=(1, 2, 3))
    union = pred_mask.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - inter
    return ((inter + smooth) / (union + smooth)).mean().item()


# ── classification ───────────────────────────────────────────────────────

def calculate_classification_metrics(
    targets, preds, probs, task_name, class_names, phase="val"
):
    """Return ``(metrics_dict, classification_report_dict)``."""
    if len(targets) == 0:
        return {}, {}

    present = sorted(set(targets) | set(preds))
    # present_names = [class_names[i] for i in present]
    present_names = [class_names[i] for i in present if i < len(class_names)]

    acc = accuracy_score(targets, preds)
    bal = balanced_accuracy_score(targets, preds)
    f1m = f1_score(targets, preds, average="macro", labels=present, zero_division=0)
    f1w = f1_score(targets, preds, average="weighted", labels=present, zero_division=0)
    mcc = matthews_corrcoef(targets, preds)

    try:
        auc = (
            roc_auc_score(targets, probs[:, 1])
            if len(present) == 2
            else roc_auc_score(targets, probs, multi_class="ovr",
                               labels=present, average="macro")
        )
    except Exception:
        auc = 0.0

    metrics = {
        f"{phase}/{task_name}_acc": acc,
        f"{phase}/{task_name}_bal_acc": bal,
        f"{phase}/{task_name}_f1_macro": f1m,
        f"{phase}/{task_name}_f1_weighted": f1w,
        f"{phase}/{task_name}_mcc": mcc,
        f"{phase}/{task_name}_auc": auc,
    }
    # Weighted Cohen's Kappa for ordinal tasks only
    if task_name in ORDINAL_TASKS:
        try:
            wck = cohen_kappa_score(targets, preds, weights="quadratic")
        except Exception:
            wck = 0.0
        metrics[f"{phase}/{task_name}_wck"] = wck
        
    report = classification_report(
        targets, preds, labels=present, target_names=present_names,
        zero_division=0, output_dict=True,
    )
    return metrics, report


# ── visualisation ────────────────────────────────────────────────────────

def plot_confusion_matrix(targets, preds, class_names, task_name):
    labels = np.arange(len(class_names))
    cm = confusion_matrix(targets, preds, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix: {task_name}")
    plt.tight_layout()
    plt.close(fig)
    return fig


def print_metrics_table(metrics_dict, task_names, phase="val"):
    """Pretty-print a table of per-task metrics."""
    print(f"\n{'Task':<15} | {'Acc':<6} | {'BalAcc':<6} | {'F1Mac':<6} | "
          f"{'F1Wt':<6} | {'MCC':<6} | {'AUC':<6} | {'WCK':<6}")
    print("-" * 75)
    for t in task_names:
        row = [metrics_dict.get(f"{phase}/{t}_{m}", 0.0)
               for m in ("acc", "bal_acc", "f1_macro", "f1_weighted", "mcc", "auc")]
        print(f"{t:<15} | " + " | ".join(f"{v:.4f}" for v in row))
    
    for t in task_names:
        acc  = metrics_dict.get(f"{phase}/{t}_acc",      0.0)
        bal_acc = metrics_dict.get(f"{phase}/{t}_bal_acc",  0.0)
        f1m  = metrics_dict.get(f"{phase}/{t}_f1_macro", 0.0)
        f1w = metrics_dict.get(f"{phase}/{t}_f1_weighted", 0.0)
        mcc  = metrics_dict.get(f"{phase}/{t}_mcc",      0.0)
        auc  = metrics_dict.get(f"{phase}/{t}_auc",      0.0)
        wck  = metrics_dict.get(f"{phase}/{t}_wck",      None)
        wck_str = f"{wck:.4f}" if wck is not None else "  —  "
        print(f"{t:<15} | {acc:<7.4f} | {bal_acc:<7.4f} | {f1m:<7.4f} | {f1w:<7.4f} | "
              f"{mcc:<7.4f} | {auc:<7.4f} | {wck_str}")
        
