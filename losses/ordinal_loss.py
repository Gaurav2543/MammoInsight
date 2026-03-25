"""
Weighted Ordinal Regression Loss.

For *K* classes the model predicts *K-1* binary logits corresponding to the
cumulative probabilities P(y > 0), P(y > 1), …, P(y > K-2).
An optional per-class weight vector handles class imbalance.
"""

import torch
import torch.nn as nn


class WeightedOrdinalRegressionLoss(nn.Module):
    """Ordinal regression with optional class-wise reweighting.

    Parameters
    ----------
    num_classes : int
        Number of ordinal ranks (e.g. 3 for Normal / Benign / Malignant).
    weights : torch.Tensor or None
        Shape ``[num_classes]``.  Each sample's loss is scaled by
        ``weights[target]``.
    """

    def __init__(self, num_classes: int, weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer(
            "weights",
            weights if weights is not None else torch.ones(num_classes),
        )
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : ``[B, K-1]``  binary logits for each cumulative threshold.
        targets : ``[B]``       integer class indices in ``[0, K-1]``.

        Returns
        -------
        loss : ``[B]`` per-sample loss (apply validity mask externally).
        """
        B = logits.shape[0]
        K = self.num_classes
        device = logits.device

        if logits.shape[1] != K - 1:
            raise ValueError(
                f"Expected {K - 1} logits for {K} classes, got {logits.shape[1]}"
            )

        # Build ordinal targets: for class c, P(y > k) = 1 when c > k
        ord_targets = torch.zeros(B, K - 1, device=device)
        for k in range(K - 1):
            ord_targets[:, k] = (targets > k).float()

        loss = self.bce(logits, ord_targets)  # [B, K-1]

        # Per-class weighting
        sample_w = self.weights[targets.long()].unsqueeze(1)  # [B, 1]
        loss = loss * sample_w

        return loss.mean(dim=1)  # [B]
