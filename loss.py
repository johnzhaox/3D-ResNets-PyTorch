import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules


class FocalLoss(nn.Module):
    """
    An implementation of Focal Loss.

    Parameters
    ----------
    alpha: 1D Tensor
        Same with torch.nn.functional.cross_entropy(), a manual rescaling weight
        given to each class. If given, has to be a Tensor of size `C`
    gamma: float=2
    reduction: str="mean"
        "mean" - averaged for each mini batch
        "sum" - summed for each mini batch

    """

    def __init__(self, alpha: torch.Tensor=None, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = None if alpha is None else alpha.float()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction=self.reduction,
                                  weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)

        if self.reduction == "mean":
            focal_loss = focal_loss.mean()
        else:
            focal_loss = focal_loss.sum()

        return focal_loss
