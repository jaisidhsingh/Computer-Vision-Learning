import torch
import numpy as np
import math
from metrics import IOU
import torch.nn as nn


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSE(reduction='sum')
        self.S = S
        self.B = B
        self.C = C
        self.LNoobj = 0.5
        self.LCoord = 5

    def forward(self, predictions, targets):
        predictions = predictions.reshap(-1, self.S, self.S, self.C+self.B*5)

        IOU_b1 = IOU(predictions[..., 21:25], targets[..., 21:25])
        IOU_b2 = IOU(predictions[..., 26:30], targets[..., 21:25])
        IOUs = torch.cat(IOU_b1.unsqueeze(0), IOU_b2.unsqueeze(0), dim=0)
        max_value, bestbox = torch.max(IOUs, dim=0)
        # I_obj for cell i (is there an object in cell i)
        exists_box = targets[..., 20].unqueeze(3)

        # ====================================================
        # BOX COORDINATE PART:
        box_predictions = exists_box*(
            bestbox*predictions[..., 26:30]  # for the second box
            # for the first box, cause b1 is first and bestbox will be 0 in that case
            + (1-bestbox)*predictions[..., 21:25]
        )

        box_targets = exists_box*targets[..., 21:25]

        box_predictions[..., 2:4] = torch.sign(
            box_predictions[..., 2:4])*torch.sqrt(torch.abs(box_predictions[..., 2:4]) + 1e-6)
        # cause derivative of sqrt 0 will be infinity, for stability during backprop

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        # maintain (N, S, S, 4) => (N*S*S, 4) cause mse error

        box_loss = self.mse(torch.flatten(
            box_predictions, end_dim=-2), torch.flatte(box_targets, end_dim=-2))

        # ====================================================
        # OBJECT PRESENCE PART:
        pred_box = bestbox*predictions[..., 25:26] + \
            (1-bestbox)*predictions[..., 20:21]
        object_loss = self.mse(torch.flatten(
            exists_box*pred_box), torch.flatten(exists_box*targets[..., 20:21]))

        # ====================================================
        # OBJECT ABSENCE PART:
        no_object_loss = self.mse(torch.flatten(
            (1-exists_box)*predictions[..., 20:21], start_dim=1), torch.flatten((1-exists_box)*targets[..., 20:21], start_dim=1))
        no_object_loss += self.mse(torch.flatten((1-exists_box)*predictions[..., 25:26], start_dim=1), torch.flatten(
            (1-exists_box)*targets[..., 20:21], start_dim=1))

        # ====================================================
        # CLASS PREDICTION PART:
        # n,s,s,20 => n*s*s, 20
        class_loss = self.mse(torch.flatten(
            exists_box*predictions[..., :20], end_dim=-2), torch.flatten(exists_box*targets[..., :20], end_dim=-2))

        loss = (
            self.LCoord*box_loss + object_loss + self.LNoobj*no_object_loss + class_loss
        )

        return loss
