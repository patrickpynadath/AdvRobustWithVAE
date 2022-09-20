from Models import LightningResnet
from Utils import Cifar10DataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

BATCH_SIZE = 100
MAX_EPOCHS = 100
DATALOADER_WORKERS = 1

if __name__ == '__main__':
    dir = '*/'
    logger = TensorBoardLogger(save_dir="lightning_logs", name="resnet_test", flush_secs = 30)
    data_module = Cifar10DataModule(data_dir=dir, batch_size=BATCH_SIZE, num_workers=DATALOADER_WORKERS)
    trainer = Trainer(gpus=2, auto_select_gpus=True, fast_dev_run=False, enable_progress_bar=True, max_epochs=MAX_EPOCHS, logger=logger)
    resnet = LightningResnet(depth=56, num_classes=10, block_name='BottleNeck')
    trainer.fit(model=resnet, datamodule=data_module)
    #trainer.test(datamodule=data_module)

