import torch.nn as nn

class TrainingWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        return self(x)["prediction"]

    def calculate_loss(self, x, target, loss_func, propagate=True):
        pred = self(x)
        loss = loss_func(pred, target)
        if propagate:
            loss.backward()
        return loss 
