"""
Source: https://github.com/adambielski/siamese-triplet/blob/master/losses.py
"""

from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9
        self.distance = lambda x, y: 1-F.cosine_similarity(x, y)

    def forward(self, output1, output2, target, size_average=True):
        # distances = (output2 - output1).pow(2).sum(1)  # squared distances
        # distances = self.distance(output1, output2)
        #losses = 0.5 * (target.float() * distances +
        #                (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        #return losses.mean() if size_average else losses.sum()
        #losses = 0.5 * (
        #            labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(self.margin - distances).pow(2))
        #return losses.mean() if self.size_average else losses.sum()
        distances = self.distance(output1, output2)
        losses = 0.5 * (
                target.float() * distances.pow(2) + (1 - target.float()).float()
                * F.relu(self.margin - distances).pow(2))
        return losses.mean() if size_average else losses.sum()
