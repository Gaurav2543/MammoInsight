"""
Multi-task mammography dataset.

Features
--------
- Backbone-specific normalisation (ImageNet vs OpenCLIP stats)
- Strict class filtering with detailed logging
- Per-sample validity masks for missing labels
- Single-view support (e.g. DMID)
"""

import os
import torch
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms


class MammoBenchDataset(Dataset):
    """Dataset for the MammoFormer multi-task pipeline.

    Parameters
    ----------
    csv_file : str
        Path to the master CSV produced by ``create_split.py``.
    split : str
        One of ``'train'``, ``'val'``, ``'test'``.
    root_dir : str
        Root directory prepended to image / mask paths.
    transform : callable, optional
        Albumentations or torchvision transform applied *after* loading.
    img_size : int
        Target spatial resolution (H = W).
    backbone_type : str
        Controls normalisation statistics.
    """

    # ── label maps (class → int) ────────────────────────────────────────
    MAP_SEVERITY = {"Normal": 0, "Benign": 1, "Malignant": 2}
    MAP_DENSITY = {"A": 0, "B": 1, "C": 2, "D": 3}
    MAP_ABNORMALITY = {"normal": 0, "mass": 1, "calcification": 2}
    MAP_MOLECULAR = {
        "Luminal A": 0,
        "Luminal B": 1,
        "HER2-enriched": 2,
        "triple negative": 3,
    }

    DATASET_STATS = {
        "cbis-ddsm":      {"mean": 0.1690, "std": 0.2327},
        "cdd-cesm":       {"mean": 0.2058, "std": 0.1866},
        "cmmd":           {"mean": 0.0599, "std": 0.1338},
        "dmid":           {"mean": 0.1395, "std": 0.2312},
        "ibia":           {"mean": 0.1013, "std": 0.1692},
        "inbreast":       {"mean": 0.1600, "std": 0.2470},
        "kau-bcmd":       {"mean": 0.0764, "std": 0.1449},
        "rsna-screening": {"mean": 0.2326, "std": 0.2002},
        "vindr-mammo":    {"mean": 0.2288, "std": 0.2267},
        "_default":       {"mean": 0.1526, "std": 0.1969},
    }

    # Global target — weighted toward your dominant training datasets
    GLOBAL_TARGET_MEAN = 0.1526  # the _default mean = global training average
    GLOBAL_TARGET_STD  = 0.1969

    CLAHE_PARAMS = {
        # Dark datasets — aggressive enhancement to bring up to global mean
        "cmmd":        {"clip_limit": 3.0, "tile_grid": (8, 8)},
        "kau-bcmd":    {"clip_limit": 2.5, "tile_grid": (8, 8)},
        "ibia":        {"clip_limit": 2.5, "tile_grid": (8, 8)},
        # Medium datasets — light enhancement
        "inbreast":    {"clip_limit": 2.0, "tile_grid": (8, 8)},
        "cbis-ddsm":   {"clip_limit": 1.5, "tile_grid": (8, 8)},
        "dmid":        {"clip_limit": 1.5, "tile_grid": (8, 8)},
        # Bright datasets already near global mean — minimal touch
        "cdd-cesm":    None,   # CESM modality — do not alter contrast
        "vindr-mammo": {"clip_limit": 1.2, "tile_grid": (8, 8)},
        "rsna-screening": {"clip_limit": 1.2, "tile_grid": (8, 8)},
        "_default":    {"clip_limit": 1.5, "tile_grid": (8, 8)},
    }

    def __init__(
        self,
        csv_file: str,
        split: str = "train",
        root_dir: str = "",
        transform=None,
        img_size: int = 224,
        backbone_type: str = "efficientnet",
    ):
        full_df = pd.read_csv(csv_file)
        self.df = full_df[full_df["split"] == split].copy()
        print(f"[{split.upper()}] Loaded {len(self.df)} rows")

        self.df = self.df.reset_index(drop=True)

        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size

        # ── normalisation stats ──────────────────────────────────────────
        if "biomedclip" in backbone_type.lower():
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=mean, std=std)

        print(
            f"[{split.upper()}] Final: {len(self.df)} cases | "
            f"Resolution: {self.img_size}×{self.img_size}\n"
        )

    # ── I/O helpers ──────────────────────────────────────────────────────

    def _resolve_path(self, path):
        """Return an absolute path or None."""
        if pd.isna(path) or path is None:
            return None
        full = os.path.join(self.root_dir, path)
        if os.path.exists(full):
            return full
        if path.startswith("/") and os.path.exists(path):
            return path
        return None
    
    @staticmethod
    def _apply_clahe(img_rgb, clip_limit, tile_grid):
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        enhanced = cv2.merge([clahe.apply(l), a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

    @staticmethod
    def _stretch_to_target(img_rgb, src_mean, src_std, tgt_mean, tgt_std):
        """
        Linearly rescale pixel intensities so the image has
        approximately the target mean and std.
        This is a per-image z-score followed by rescaling — 
        it makes dark datasets look more like the global distribution.
        """
        t = img_rgb.astype(np.float32) / 255.0
        # z-score using dataset stats, then rescale to global target
        t_normalized = (t - src_mean) / (src_std + 1e-6)
        t_rescaled = t_normalized * tgt_std + tgt_mean
        t_clipped = np.clip(t_rescaled, 0.0, 1.0)
        return (t_clipped * 255).astype(np.uint8)

    def _load_image(self, path, source_dataset="_default"):
        empty = torch.zeros(3, self.img_size, self.img_size, dtype=torch.float32)
        resolved = self._resolve_path(path)
        if resolved is None:
            return empty, False

        img = cv2.imread(resolved)
        if img is None:
            return empty, False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        ds_key = source_dataset.lower()

        # Step 1: CLAHE (skip for CESM and already-bright datasets)
        clahe_p = self.CLAHE_PARAMS.get(ds_key, self.CLAHE_PARAMS["_default"])
        if clahe_p is not None:
            img = self._apply_clahe(img, clahe_p["clip_limit"], clahe_p["tile_grid"])

        # Step 2: Stretch to global distribution
        ds_stats = self.DATASET_STATS.get(ds_key, self.DATASET_STATS["_default"])
        img = self._stretch_to_target(
            img,
            src_mean=ds_stats["mean"],
            src_std=ds_stats["std"],
            tgt_mean=self.GLOBAL_TARGET_MEAN,
            tgt_std=self.GLOBAL_TARGET_STD,
        )

        # Step 3: Standard ImageNet normalization (unchanged — model expects this)
        t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        return self.normalize(t), True

    def _load_mask(self, path):
        empty = torch.zeros(1, self.img_size, self.img_size, dtype=torch.float32)
        resolved = self._resolve_path(path)
        if resolved is None:
            return empty, False
        mask = cv2.imread(resolved, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return empty, False
        if self.transform is None:
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(np.float32)
            return torch.from_numpy(mask).unsqueeze(0), True
        return mask, True

    # ── label extraction ─────────────────────────────────────────────────

    @staticmethod
    def _get_label(value, mapping, default=0):
        """Map a raw value to an integer label, returning (label, validity)."""
        if pd.notna(value):
            key = str(value).strip()
            for mk, mv in mapping.items():
                if key.lower() == mk.lower():
                    return mv, 1.0
        return default, 0.0

    # ── main entry ───────────────────────────────────────────────────────

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        src = str(row.get("source_dataset", "_default"))
        cc_img, has_cc   = self._load_image(row.get("cc_path"),  source_dataset=src)
        mlo_img, has_mlo = self._load_image(row.get("mlo_path"), source_dataset=src)
        cc_mask, has_cc_mask = self._load_mask(row.get("cc_roi"))
        mlo_mask, has_mlo_mask = self._load_mask(row.get("mlo_roi"))

        view_mask = torch.tensor(
            [1.0 if has_cc else 0.0, 1.0 if has_mlo else 0.0], dtype=torch.float32
        )
        has_seg = 1.0 if (has_cc_mask or has_mlo_mask) else 0.0

        labels, validity = {}, {}

        labels["classification"], validity["classification"] = self._get_label(
            row.get("classification"), self.MAP_SEVERITY
        )
        labels["density"], validity["density"] = self._get_label(
            row.get("density"), self.MAP_DENSITY
        )
        labels["abnormality"], validity["abnormality"] = self._get_label(
            row.get("abnormality"), self.MAP_ABNORMALITY
        )
        labels["molecular"], validity["molecular"] = self._get_label(
            row.get("molecular_subtype"), self.MAP_MOLECULAR
        )

        # BI-RADS (numeric 0-5)
        labels["birads"], validity["birads"] = 0, 0.0
        if pd.notna(row.get("birads")):
            try:
                b = int(float(row["birads"]))
                if 0 <= b <= 5:
                    labels["birads"], validity["birads"] = b, 1.0
            except (ValueError, TypeError):
                pass

        # Age (normalised to [0, 1])
        age = row.get("age")
        if pd.notna(age) and float(age) > 0:
            labels["age"], validity["age"] = float(age) / 100.0, 1.0
        else:
            labels["age"], validity["age"] = 0.0, 0.0

        return {
            "cc_image": cc_img,
            "mlo_image": mlo_img,
            "view_mask": view_mask,
            "cc_mask": cc_mask,
            "mlo_mask": mlo_mask,
            "has_seg": has_seg,
            "labels": labels,
            "validity": validity,
            "breast_id": str(row["breast_id"]),
            "source_dataset": str(row.get("source_dataset", "unknown")),
        }
