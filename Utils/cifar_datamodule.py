from torch.utils.data import DataLoader
from Utils.utils import get_cifar_sets
import pytorch_lightning as pl


class Cifar10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir : str, batch_size : int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage = None):
        self.cifar_train, self.cifar_test = get_cifar_sets(self.data_dir)


    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.cifar_test, batch_size=self.batch_size, shuffle=False)