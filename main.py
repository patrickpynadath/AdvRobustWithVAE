from Models import LightningResnet
from Utils import Cifar10DataModule
from pytorch_lightning import Trainer

BATCH_SIZE = 100
MAX_EPOCHS = 100

if __name__ == 'main':
    dir = '*/'
    data_module = Cifar10DataModule(data_dir=dir, batch_size=BATCH_SIZE)
    trainer = Trainer(gpus=2, auto_select_gpus=True, fast_dev_run=True, enable_progress_bar=True, max_epochs=MAX_EPOCHS)
    resnet = LightningResnet(depth=20, num_classes=10, block_name='BasicBlock')
    trainer.fit(model=resnet, datamodule=data_module)
    trainer.test()


