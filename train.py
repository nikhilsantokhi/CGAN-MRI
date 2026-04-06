# Import necessary packages
from pl_modules import OCMRDataModule, CGANModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import transforms
from model import Generator, Discriminator, Discriminator_v2
from callbacks import ModelSummaryLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn as nn
import torch

# optimise mat mult precision to improve performance
torch.set_float32_matmul_precision('medium')

g = Generator()
d = Discriminator()

# initialise weights
def init_weights(m):
    if hasattr(m, 'weight') and m.weight is not None:
        nn.init.normal_(m.weight, 0.0, 0.02)

    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias, 0)

g.apply(init_weights)
d.apply(init_weights)

# load loggers
tb_logger = TensorBoardLogger(
    save_dir='checkpoints',
)

checkpoint = ModelCheckpoint(
    monitor='val/g_loss',
    mode='min',
    save_top_k=1,
    filename='cgan-{epoch:03d}-{val_g_loss:.4f}',
)

summary_logger = ModelSummaryLogger()

# instantiate pytorch lightning classes
dm = OCMRDataModule(
    dataset_dir='./datasets/OCMR/',
    acceleration='R=8',
    batch_size=16,
    num_workers=4,
    train_transform=transforms.ToTensor(),
    val_transform=transforms.ToTensor(),
    test_transform=transforms.ToTensor(),
)

cgan_module = CGANModule(discriminator=d, generator=g)

# define the trainer
trainer = pl.Trainer(
    max_epochs=50,
    logger=tb_logger,
    callbacks=[checkpoint, summary_logger]
)

# train
def main():
    trainer.fit(model=cgan_module, datamodule=dm)
    cgan_module.save_model('saved_models')


if __name__ == '__main__':
    main()