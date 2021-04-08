import torch
import torch.nn as nn

class DictWrapper(nn.Module):
    def __init__(self, loss_dict, weights_dict={"prediction": 1.0}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights_dict = weights_dict
        self.loss_dict = loss_dict

    def forward(self, x, target):
        assert isinstance(x, dict)
        assert isinstance(target, torch.Tensor)

        losses = []
        for key, pred in x.items():
            if key in self.loss_dict.keys():
                assert pred.device == target.device
                losses.append(self.loss_dict[key](pred, target) * self.weights_dict[key])

        return torch.stack(losses).mean()
