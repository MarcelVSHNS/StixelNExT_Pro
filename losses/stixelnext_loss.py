from torch import nn
import torch


class StixelLoss(nn.Module):
    # threshold means the threshold when (probab) a border is detected
    def __init__(self, alpha=False, beta=False, gamma=False):
        """Intersection over Union"""
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.bce_loss = nn.BCELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss_bce_occ = 0.0
        loss_maximum_cuts_occ = 0.0
        loss_bce_cut = 0.0
        single_monitoring_dicts = []
        if self.alpha:
            loss_bce_occ = self.bce_loss(inputs[:, 0, :, :], targets[:, 0, :, :])
            loss_bce_occ = self.alpha * loss_bce_occ
            single_monitoring_dicts.append({'name': "bce_occupancy", 'value': loss_bce_occ})
        if self.beta:
            loss_maximum_cuts_occ = torch.mean(inputs[:, 0, :, :])
            loss_maximum_cuts_occ = self.beta * loss_maximum_cuts_occ
            single_monitoring_dicts.append({'name': "sum_occupancy", 'value': loss_maximum_cuts_occ})
        if self.gamma:
            loss_bce_cut = self.bce_loss(inputs[:, 1, :, :], targets[:, 1, :, :])
            loss_bce_cut = self.alpha * loss_bce_cut
            single_monitoring_dicts.append({'name': "bce_edges", 'value': loss_bce_cut})
        return loss_bce_occ + loss_maximum_cuts_occ + loss_bce_cut, single_monitoring_dicts
