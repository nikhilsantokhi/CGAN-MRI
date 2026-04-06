# Import necessary packages
import pytorch_lightning as pl
import torch
from torch import nn
import os
import torchvision

from losses import generator_loss, discriminator_loss

class CGANModule(pl.LightningModule):
    def __init__(
            self,
            generator: nn.Module,
            discriminator: nn.Module,
            lr_g: float = 0.0002,
            lr_d: float = 0.0002,
            beta1: float = 0.5,
            beta2: float = 0.999,
            alpha: float = 0.1,
            sigma: float = 100.0,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=['generator', 'discriminator'])

        self.generator = generator
        self.discriminator = discriminator

        self.lr_g = lr_g
        self.lr_d = lr_d
        self.beta1 = beta1
        self.beta2 = beta2

        self.alpha = alpha
        self.sigma = sigma

        self.automatic_optimization = False

    def forward(self, x):
        return self.generator(x)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr_g,
                                 betas=(self.beta1, self.beta2))

        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_d,
                                 betas=(self.beta1, self.beta2))

        return [opt_g, opt_d]

    def training_step(self, batch, batch_idx):
        x, y = batch

        opt_g, opt_d = self.optimizers()

        # train discriminator
        d_loss = self.discriminator_training_step(x, y)

        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        # train generator
        g_loss = self.generator_training_step(x, y)

        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        # log losses
        self.log_dict({
            'train/d_loss': d_loss,
            'train/g_loss': g_loss,
        },
            prog_bar=True,
            on_epoch=True,
            on_step=True,
        )

    def discriminator_training_step(self, x, y):
        self.discriminator.train()
        self.generator.eval()

        y_fake = self.generator(x)
        d_loss = discriminator_loss(self.discriminator, x, y, y_fake)

        return d_loss

    def generator_training_step(self, x, y):
        self.discriminator.eval()
        self.generator.train()

        y_fake = self.generator(x)
        g_loss = generator_loss(self.discriminator, x, y, y_fake, self.alpha, self.sigma)

        return g_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_fake = self.generator(x)

        g_loss = generator_loss(self.discriminator, x, y, y_fake, self.alpha, self.sigma)

        # log loss
        self.log(
            'val/g_loss',
            g_loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # store first batch only
        if batch_idx == 0:
            aliased_img = x[:4]
            fake_img = y_fake[:4]
            gt_img = y[:4]
            composite = torch.cat([aliased_img, fake_img, gt_img], dim=0)
            grid = torchvision.utils.make_grid(composite, nrow=4, pad_value=1)
            self.logger.experiment.add_image('validation images', grid, self.current_epoch)

            # return values for further use in validation_epoch_end

        return {'val_loss': g_loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_fake = self.generator(x)

        g_loss = generator_loss(self.discriminator, x, y, y_fake, self.alpha, self.sigma)

        # log loss
        self.log(
            'test/g_loss',
            g_loss,
            on_epoch=True,
        )

    def save_model(self, path):
        torch.save(self.generator.state_dict(), os.path.join(path, 'generator.pth'))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('CGAN')

        parser.add_argument('--lr_g', type=float, default=0.0002)
        parser.add_argument('--lr_d', type=float, default=0.0002)
        parser.add_argument('--beta1', type=float, default=0.5)
        parser.add_argument('--beta2', type=float, default=0.999)
        parser.add_argument('--alpha', type=float, default=0.1)
        parser.add_argument('--sigma', type=float, default=100.0)

        return parent_parser
