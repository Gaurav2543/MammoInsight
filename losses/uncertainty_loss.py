# import torch
# import torch.nn as nn

# class MultiTaskUncertaintyLoss(nn.Module):
#     """
#     Kendall et al. (2018) homoscedastic uncertainty weighting.
    
#     For regression/ordinal tasks:   L_i / (2 * sigma_i^2) + log(sigma_i)
#     For classification tasks:       L_i / sigma_i^2 + log(sigma_i)
    
#     We parameterise as log_var_i = log(sigma_i^2) for numerical stability.
#     task_types: dict mapping task name -> "regression" or "classification"
#     """
#     def __init__(self, task_names, task_types):
#         super().__init__()
#         self.task_names = task_names
#         self.task_types = task_types  # {"severity": "regression", "abnormality": "classification", ...}
#         # Initialise all log-variances to 0 (sigma=1, no scaling initially)
#         self.log_vars = nn.ParameterDict({
#             t: nn.Parameter(torch.zeros(1)) for t in task_names
#         })

#     def forward(self, losses: dict):
#         """
#         losses: dict of {task_name: scalar loss tensor}
#         Returns: (total_loss, {task_name: effective_weight}) for logging
#         """
#         total = torch.tensor(0.0, device=next(iter(losses.values())).device)
#         weights = {}

#         for task, loss in losses.items():
#             # lv = self.log_vars[task]
#             lv = self.log_vars[task].clamp(-4.0, 4.0)  
#             if self.task_types.get(task, "classification") == "regression":
#                 # ordinal / regression branch
#                 precision = torch.exp(-lv)
#                 total = total + 0.5 * precision * loss + 0.5 * lv
#             else:
#                 # classification branch
#                 precision = torch.exp(-lv)
#                 total = total + precision * loss + lv
#             weights[task] = torch.exp(-lv).item()

#         return total, weights

import torch
import torch.nn as nn

class MultiTaskUncertaintyLoss(nn.Module):
    """
    Kendall et al. (2018) homoscedastic uncertainty weighting.
    
    For regression/ordinal tasks:   L_i / (2 * sigma_i^2) + log(sigma_i)
    For classification tasks:       L_i / sigma_i^2 + log(sigma_i)
    
    We parameterise as log_var_i = log(sigma_i^2) for numerical stability.
    task_types: dict mapping task name -> "regression" or "classification"
    """
    def __init__(self, task_names, task_types):
        super().__init__()
        self.task_names = task_names
        self.task_types = task_types  # {"severity": "regression", "abnormality": "classification", ...}
        # Initialise all log-variances to 0 (sigma=1, no scaling initially)
        self.log_vars = nn.ParameterDict({
            t: nn.Parameter(torch.zeros(1)) for t in task_names
        })

    def forward(self, losses: dict):
        """
        losses: dict of {task_name: scalar loss tensor}
        Returns: (total_loss, {task_name: effective_weight}) for logging
        """
        total = torch.tensor(0.0, device=next(iter(losses.values())).device)
        weights = {}

        for task, loss in losses.items():
            lv = self.log_vars[task]
            if self.task_types.get(task, "classification") == "regression":
                # ordinal / regression branch
                precision = torch.exp(-lv)
                total = total + 0.5 * precision * loss + 0.5 * lv
            else:
                # classification branch
                precision = torch.exp(-lv)
                total = total + precision * loss + lv
            weights[task] = torch.exp(-lv).item()

        return total, weights