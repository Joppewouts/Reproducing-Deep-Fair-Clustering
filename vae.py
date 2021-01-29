import argparse

import numpy as np

import torch
import wandb
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from dataloader import get_dataset
from kmeans import get_cluster_centers
from module import Encoder
from utils import init_weights
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, Callback
from pl_bolts.models.autoencoders import VAE
from sklearn.cluster import KMeans
from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
    resnet50_decoder,
    resnet50_encoder,
)
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
from argparse import ArgumentParser


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, size):
        super(UnFlatten, self).__init__()
        self.size = size

    def forward(self, input):
        return input.view(input.size(0), *self.size)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.seq = nn.Sequential(
            nn.Linear(64, 512),  # h_dim to z_dim
            nn.ReLU(),
            nn.Linear(512, 8 * 8 * 16),  # (B, 1024)
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            UnFlatten([16, 8, 8]),  # (B, 16, 8, 8)
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            # (B, 32, 16, 16)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),  # (B, 32, 16, 16)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            # (B, 16, 32, 32)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1, bias=False),  # (B, 1, 32, 32)
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.1)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)

    def forward(self, x):
        return self.seq(x)

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1}]


def dfc_encoder(*args):
    return Encoder()


def dfc_decoder(*args):
    return Decoder()


class DFC_VAE(pl.LightningModule):
    def __init__(
            self,
            input_height: int,
            enc_type: str = 'resnet18',
            first_conv: bool = False,
            maxpool1: bool = False,
            enc_out_dim: int = 512,
            latent_dim: int = 256,
            lr: float = 1e-4,
            log_name: str = 'dfc_vae',
            batch_size: int = None,
            dataset_length: int = None,
            **kwargs
    ):
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            kl_coeff: coefficient for kl term of the loss
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super(DFC_VAE, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height
        self.log_name = log_name
        self.batch_size = batch_size
        self.dataset_length = dataset_length

        self.encoder = dfc_encoder(first_conv, maxpool1)
        self.decoder = dfc_decoder(self.latent_dim, self.input_height, first_conv, maxpool1)

    def forward(self, x):
        z, _, _ = self.encoder(x)
        return self.decoder(z)

    def _run_step(self, x):
        z, mu, log_var = self.encoder(x)
        # p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), mu, log_var

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):
        x, y = batch
        z, x_hat, mu, log_var = self._run_step(x)

        recon_loss = F.mse_loss(x_hat, x, reduction='mean')

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1),
                              dim=0) * self.batch_size / self.dataset_length

        loss = kld_loss + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kld_loss,
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict(
            {f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False
        )
        logs['batch'] = batch_idx
        wandb.log({f"{self.log_name}_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def train(args, log_name, dataloader_list, input_height, is_digit_dataset=True, device='cpu',
          encoder_pretrain_path=None):
    """This function trains a variational autoencoder given a dataloader.

    Args:
        args: general arguments including: log_dir, seed, encoder_lr, encoder_bs, encoder_max_epochs, log_dir
        log_name: prefix for logging files
        dataloader_list: list of one or more dataloaders used as training data
        input_height: entropy_input height of the images
        is_digit_dataset: if true the custom encoder is used, otherwise resnet50 based on imagenet
        device: cpu/gpu
        encoder_pretrain_path (str): provide the path to the pretained encoder

    Returns:
        DFC_VAE model
    """

    pl.seed_everything(seed=args.seed)

    concat_dataset = torch.utils.data.ConcatDataset([x.dataset for x in dataloader_list])
    dataloader = torch.utils.data.DataLoader(
        dataset=concat_dataset,
        batch_size=args.encoder_bs,
        shuffle=True,
        num_workers=4
    )

    if is_digit_dataset:
        model = DFC_VAE(input_height, enc_type='dfc', latent_dim=64, enc_out_dim=512, lr=args.encoder_lr,
                        log_name=log_name, dataset_length=len(concat_dataset), batch_size=args.encoder_bs).to(device)
    else:
        raise NotImplementedError

    if encoder_pretrain_path is not None:
        model.encoder.load_state_dict(torch.load(encoder_pretrain_path, map_location=device))

    wandb.watch(model)

    checkpoint_callback = ModelCheckpoint(monitor='train_loss', filepath=args.log_dir + log_name + "/", verbose=True)
    if device.type == 'cpu':
        trainer = pl.Trainer(
            checkpoint_callback=checkpoint_callback,
            progress_bar_refresh_rate=1,
            max_epochs=args.encoder_max_epochs
        )
    else:
        trainer = pl.Trainer(
            checkpoint_callback=checkpoint_callback,
            progress_bar_refresh_rate=1,
            max_epochs=args.encoder_max_epochs,
            gpus=1 if device != 'cpu' else None
        )
    trainer.fit(model, dataloader)
    print(
        f"Best model (loss: {checkpoint_callback.best_model_score:.3f}) stored at {checkpoint_callback.best_model_path}")
    wandb.run.summary[f"{log_name}_loss"] = checkpoint_callback.best_model_score

    model = DFC_VAE.load_from_checkpoint(checkpoint_callback.best_model_path)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--bs", type=int, default=512)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--encoder_lr", type=float, default=1e-4)
    parser.add_argument("--encoder_bs", type=int, default=128)
    parser.add_argument("--encoder_max_epochs", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="mnist_usps")
    parser.add_argument("--log_dir", type=str, default="./DFC_LOGS/")

    parser.add_argument("--cluster_n_init", type=int, default=20)
    parser.add_argument("--cluster_max_step", type=int, default=5000)
    parser.add_argument("--cluster_number", type=int, default=10)
    parser.add_argument("--cluster_path", type=str)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    dataloader_0, dataloader_1 = get_dataset[args.dataset](args)

    wandb.init(project="dfc", config=args)

    model = train(args, "encoder", [dataloader_0], 32, device=device, is_digit_dataset=True)
    encoder = model.encoder

    cluster_centers = get_cluster_centers(args, encoder, args.cluster_number, [dataloader_0],
                                          args.cluster_path, device=device, save_name="clusters_dfc")

    model.eval()
    plt.imshow(model.decoder(torch.rand(1, 64)).squeeze().detach().numpy(), cmap='gray', vmin=0, vmax=1)
    plt.show()
