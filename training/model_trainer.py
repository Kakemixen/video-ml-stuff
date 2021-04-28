import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from utils import io_utils
import os

class ModelTrainer:
    def __init__(self, model, dataloader_train, dataloader_val, dataloader_tb,
            criterion, cpt_dir="storage/current/checkpoints/", tb_dir="storage/current/tensorboard/",
            training_logs_per_epoch = 10
        ):
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.criterion = criterion 
        self.model = model
        self.optimizer = torch.optim.Adam(
                self.model.model.parameters(), 
                lr=0.0005)
        self.step = 0
        self.cpt_dir = cpt_dir
        self.dataloader_tb = dataloader_tb
        self.tb_dir = tb_dir
        self.tb = SummaryWriter(log_dir=tb_dir)
        self.log_freq = len(dataloader_train) // training_logs_per_epoch
        io_utils.make_folder_structure(self.cpt_dir, clean=True)
        io_utils.make_folder_structure(self.tb_dir, clean=True)

    def train(self, epochs=10):
        for e in range(epochs):
            print(f"epoch {e}/{epochs}")
            self.train_epoch()
            val_loss = self.validate_epoch()
            self.store_cpt(e, val_loss)
            self.tb.add_scalar("validation/loss", val_loss, self.step)
            self.visualize_data(self.dataloader_tb)

    def train_epoch(self):
        log_loss = 0
        with tqdm(self.dataloader_train) as epoch_iter:
            for i, batch in enumerate(epoch_iter):
                self.optimizer.zero_grad()
                batch_loss = self.model.calculate_loss(
                        batch, self.criterion, propagate=True)
                self.optimizer.step()
                self.step += 1
                epoch_iter.set_postfix(loss=batch_loss.item())
                log_loss += batch_loss
                if (i+1)%self.log_freq == 0:
                    self.tb.add_scalar("training/loss", log_loss/self.log_freq, self.step)
                    log_loss = 0

    def validate_epoch(self):
        val_loss = 0
        for batch in tqdm(self.dataloader_val):
            batch_loss = self.model.calculate_loss(
                    batch, self.criterion, propagate=False)
            val_loss += batch_loss
        print(f"val loss: {val_loss / len(self.dataloader_val)}")
        return val_loss / len(self.dataloader_val)

    def store_cpt(self, epoch, loss_val):
        storage_dict = {
                "model_state_dict": self.model.model.state_dict(),
                "optim_state_dict": self.optimizer.state_dict(),
                "epoch": epoch,
                "step": self.step,
                "loss": loss_val
        }
        torch.save(storage_dict, os.path.join(self.cpt_dir, f"checkpoint_{epoch}.pth.tar"))

    def visualize_data(self, dataloader):
        # TODO make selection consistent
        for num, batch in enumerate(tqdm(dataloader)):
            vids = make_overlay(batch["input"], batch["segmentation"])
            preds = make_overlay(batch["input"], self.model.predict(batch["input"]) )
            vid_grids = [make_grid([make_grid(v), make_grid(p)], nrow=1) for v,p in zip(vids, preds)]
            for i, vid_grid in enumerate(vid_grids):
                self.tb.add_image(f"sample/{i*num + i}", vid_grid, self.step)

    # test functions
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

MAPPING = torch.randint(0, 255, (41, 3)).numpy() / 255
MAPPING[0] = [0,0,0]
def make_overlay(inp, seg):
    seg_rgb = torch.Tensor(MAPPING[seg.squeeze(2)]).permute(0,1,4,2,3)
    seg_a = 0.4
    overlay = (1-seg_a)*inp + seg_a*seg_rgb
    return overlay


