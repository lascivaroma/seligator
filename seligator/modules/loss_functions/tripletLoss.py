# CC https://raw.githubusercontent.com/PonteIneptique/jdnlp/master/jdnlp/modules/loss_functions/triplet.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    https://discuss.pytorch.org/t/triplet-loss-in-pytorch/30634
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self,
                anchor: torch.Tensor,
                positive: torch.Tensor,
                negative: torch.Tensor,
                size_average: bool = True
                )\
            -> torch.Tensor:
        distance_positive = (anchor - positive).pow(2).sum(1)  # Square Distance
        distance_negative = (anchor - negative).pow(2).sum(1)  # Square Distance
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()
