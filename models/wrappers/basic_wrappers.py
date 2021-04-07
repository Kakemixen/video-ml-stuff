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
        return loss.detach()

class VideoGeneratorWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        for frame in x:
            yield self.model(frame)

    def predict(self, x):
        for frame in self(x):
            yield frame["prediction"]

    def calculate_loss(self, inputs, targets, loss_func, propagate=True):
        # transpose input and target
        inputs = inputs.permute(1,0,2,3,4)
        targets = targets.permute(1,0,2,3,4)

        pred_gen = self(inputs)
        loss = 0
        for pred, target in zip(pred_gen, targets):
            batch_loss = loss_func(pred, target)
            if propagate:
                batch_loss.backward()
            loss += batch_loss.detach()
        return loss 
