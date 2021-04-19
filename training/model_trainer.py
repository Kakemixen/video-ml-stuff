import torch
from tqdm import tqdm

class ModelTrainer:
    def __init__(self, model, dataloader_train, dataloader_val, criterion):
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.criterion = criterion 
        self.model = model
        self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=0.0005)

    def train(self, epochs=10):
        val_loss = self.validate_epoch()
        for e in range(epochs):
            print(f"epoch {e}/{epochs}")
            self.train_epoch()
            val_loss = self.validate_epoch()

    def train_epoch(self):
        with tqdm(self.dataloader_train) as epoch_iter:
            for batch in epoch_iter:
                self.optimizer.zero_grad()
                batch_loss = self.model.calculate_loss(
                        batch, self.criterion, propagate=True)
                self.optimizer.step()
                epoch_iter.set_postfix(loss=batch_loss.item())

    def validate_epoch(self):
        val_loss = 0
        for batch in tqdm(self.dataloader_val):
            batch_loss = self.model.calculate_loss(
                    batch, self.criterion, propagate=False)
            val_loss += batch_loss
        print(f"val loss: {val_loss / len(self.dataloader_val)}")
        return val_loss / len(self.dataloader_val)
    
    def overfit_batch(self):
        batch = next(iter(self.dataloader_train))
        print(f"overfitting on batch ({batch['input'].shape})")
        batch_loss = 1
        with tqdm() as bar:
            while batch_loss > 0.001:
                self.optimizer.zero_grad()
                bar.update()
                batch_loss = self.model.calculate_loss(
                        batch, self.criterion, propagate=True)
                self.optimizer.step()
                bar.set_postfix(loss=batch_loss.item())

    def train_empty_epoch(self):
        with tqdm(self.dataloader_train) as epoch_iter:
            for batch in epoch_iter:
                self.optimizer.zero_grad()
                batch["input"] = torch.zeros_like(batch["input"])
                batch_loss = self.model.calculate_loss(
                        batch, self.criterion, propagate=True)
                self.optimizer.step()
                epoch_iter.set_postfix(loss=batch_loss.item())
