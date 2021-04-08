import torch
from tqdm import tqdm

class ModelTrainer:
    def __init__(self, model, dataloader_train, dataloader_val, criterion):
        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.criterion = criterion 

    def train(self, epochs=10):
        for e in range(epochs):
            print(f"epoch {e}/{epochs}")
            self.train_epoch()
            val_loss = self.validate_epoch()
            print("val loss: {val_loss}")

    def train_epoch(self):
        curr_loss = -1
        for batch in tqdm(self.dataloader_train):
            batch_loss = self.model.calculate_loss(
                    batch, self.criterion, propagate=True)
            curr_loss = batch_loss

    def validate_epoch(self):
        val_loss = 0
        for batch in tqdm(self.dataloader_val):
            batch_loss = self.model.calculate_loss(
                    batch, self.criterion, propagate=False)
            val_loss += batch_loss
        return val_loss / len(self.dataloader_val)
