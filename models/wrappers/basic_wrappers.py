import torch
import torch.nn as nn

class TrainingWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.cuda()

    def forward(self, x):
        return self.model(x.cuda())

    def predict(self, x):
        return self(x.cuda())["prediction"]

    def calculate_loss(self, batch, loss_func, propagate=True):
        pred = self(batch["input"]) # implicit cuda()
        loss = loss_func(pred, batch["segmentation"].cuda())
        if propagate:
            loss.backward()
        return loss.detach()

class VideoGeneratorWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.cuda()

    def forward(self, x):
        for frame in x:
            yield self.model(frame)

    def predict(self, x):
        inputs = x.permute(1,0,2,3,4).cuda()
        preds = torch.stack([frame["prediction"].cpu() for frame in self(inputs)], dim=1)\
                .argmax(dim=2).unsqueeze(2)
        return preds

    def calculate_loss(self, batch, loss_func, propagate=True):
        # transpose input and target
        inputs = batch["input"].permute(1,0,2,3,4).cuda()
        targets = batch["segmentation"].permute(1,0,2,3,4).cuda()

        pred_gen = self(inputs)
        loss = 0
        for pred, target in zip(pred_gen, targets):
            batch_loss = loss_func(pred, target)
            if propagate:
                batch_loss.backward()
            loss += batch_loss.detach()
        return loss / len(inputs)
