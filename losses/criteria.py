import torch
import torch.nn as nn
import torch.nn.functional as F

class BCELoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, pred, target):
        assert pred.dim() == 4
        return F.binary_cross_entropy(pred, target, reduction="none").mean((1,2,3))

class CELoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = torch.Tensor([0.25, *[1.0]*40]).cuda()

    def forward(self, pred, target):
        assert pred.dim() == 4
        return F.cross_entropy(pred, target[:,0,:,:].type(torch.long), weight=self.weight, reduction="none").mean((1,2))

