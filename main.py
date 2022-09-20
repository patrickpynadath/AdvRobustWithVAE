from Models import LightningResnet
from torch.utils.data import DataLoader
from Utils import Cifar10DataModule, get_cifar_sets
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

BATCH_SIZE = 100
MAX_EPOCHS = 100

if __name__ == '__main__':
    dir = '*/'
    logger = TensorBoardLogger(save_dir="lightning_logs", name="resnet_test")
    data_module = Cifar10DataModule(data_dir=dir, batch_size=BATCH_SIZE)
    trainer = Trainer(gpus=1, auto_select_gpus=True, fast_dev_run=False, enable_progress_bar=True, max_epochs=MAX_EPOCHS, logger=logger, log_every_n_steps=1)
    resnet = LightningResnet(depth=20, num_classes=10, block_name='BasicBlock')
    trainer.fit(model=resnet, datamodule=data_module)
    #trainer.test(datamodule=data_module)

