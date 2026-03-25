"""
DHN-NCE: Decoupled Hard-Negative Noise Contrastive Estimation.

Task-specific projection heads map the shared 768-d combined feature to a
lower-dimensional hypersphere where same-class samples are pulled together
and hard negatives are reweighted to push them apart.

Reference: https://arxiv.org/abs/2301.02280
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DHN_NCE_Loss(nn.Module):
    """Supervised hard-negative contrastive loss with task-specific projectors.

    Parameters
    ----------
    temperature : float
        Softmax temperature (lower → sharper).
    beta : float
        Hard-negative reweighting strength (higher → focus on hard negatives).
    alpha : float
        Positive-term weight in the denominator.  Set ≥ 1.0 to guarantee
        non-negative loss.
    projection_dim : int
        Dimension of the task-specific embedding space.
    feature_dim : int
        Input feature dimension (default 768 = fusion + age).
    task_names : list[str]
        Task identifiers for which projectors are created.
    """

    DEFAULT_TASKS = ["severity", "density", "abnormality", "molecular", "birads"]

    def __init__(
        self,
        temperature: float = 0.07,
        beta: float = 0.75,
        alpha: float = 1.0,
        projection_dim: int = 128,
        feature_dim: int = 768,
        task_names: list = None,
    ):
        super().__init__()
        self.temperature = temperature
        self.beta = beta
        self.alpha = alpha

        tasks = task_names or self.DEFAULT_TASKS
        self.projectors = nn.ModuleDict(
            {t: self._make_projector(feature_dim, projection_dim) for t in tasks}
        )

    @staticmethod
    def _make_projector(in_dim: int, out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        validity: torch.Tensor,
        task_name: str,
    ) -> torch.Tensor:
        """Compute DHN-NCE for one task.

        Parameters
        ----------
        features  : ``[B, D]`` combined features.
        labels    : ``[B]``    class indices.
        validity  : ``[B]``    1 = valid, 0 = missing.
        task_name : str        key into ``self.projectors``.

        Returns
        -------
        Scalar loss (0 if fewer than 2 valid samples).
        """
        mask = validity.bool()
        if mask.sum() < 2:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        feats = features[mask]
        labs = labels[mask].long()
        B = feats.shape[0]

        # Project & normalise
        z = self.projectors[task_name](feats)       # [B, P]
        z = F.normalize(z, p=2, dim=1)

        logits = z @ z.T / self.temperature          # [B, B]

        # Positive / negative masks
        eq = labs.unsqueeze(1) == labs.unsqueeze(0)   # [B, B]
        pos_mask = eq.float().fill_diagonal_(0)
        neg_mask = (~eq).float()

        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        exp_logits = torch.exp(logits)

        # Positive term
        pos = exp_logits * pos_mask

        # Hard-negative reweighting
        neg_exp = exp_logits * neg_mask
        N = neg_mask.sum(dim=1, keepdim=True).clamp(min=1)
        norm = neg_exp.sum(dim=1, keepdim=True) + 1e-8
        reweight = N * torch.exp(self.beta * logits * neg_mask) / norm
        neg = reweight * neg_exp

        pos_sum = pos.sum(dim=1, keepdim=True) + 1e-8
        neg_sum = neg.sum(dim=1, keepdim=True) + 1e-8

        denom = self.alpha * pos_sum + neg_sum
        denom = denom.clamp(min=1e-8)

        loss = -torch.log(pos_sum / denom)
        loss = loss.clamp(min=0.0, max=10.0)

        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Dice loss for segmentation
# ─────────────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        preds   : ``[B, 1, H, W]`` logits.
        targets : ``[B, 1, H, W]`` binary masks.
        """
        probs = torch.sigmoid(preds).view(preds.shape[0], -1)
        tgt = targets.view(targets.shape[0], -1)
        inter = (probs * tgt).sum(1)
        union = probs.sum(1) + tgt.sum(1)
        dice = (2.0 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()
