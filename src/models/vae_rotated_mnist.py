# RotatedMNIST model specification

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

from numpy import prod, sqrt
from .vae import VAE
from utils import Constants
from distributions.sparse import Sparse
from torch.distributions import Laplace, Normal

# Constants
dataSize = torch.Size([1, 28, 28])
imgChans = dataSize[0]
data_dim = int(prod(dataSize))

class Enc(nn.Module):
    def __init__(self, params):
        super(Enc, self).__init__()
        self.enc = nn.Sequential(
            # input size is in_size x 64 x 64
            nn.Conv2d(imgChans, params.fBase, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: ndf x 32 x 32
            nn.Conv2d(params.fBase, params.fBase * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(params.fBase * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf * 2) x 16 x 16
            nn.Conv2d(params.fBase * 2, params.fBase * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(params.fBase * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf * 4) x 8 x 8
        )
        self.c1 = nn.Conv2d(params.fBase * 4, params.latent_dim, 4)
        self.c2 = nn.Conv2d(params.fBase * 4, params.latent_dim, 4)
        # c1, c2 size: latent_dim x 1 x 1

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        pre_shape = x.size()[:-3]
        e = self.enc(x.view(-1, *torch.Size([1, 32, 32])))
        mu = self.c1(e).view(*pre_shape, -1)
        scale = F.softplus(self.c2(e)).view_as(mu) + Constants.eta
        return mu, scale

class Dec(nn.Module):
    def __init__(self, params):
        super(Dec, self).__init__()
        self.dec = nn.Sequential(
            # input size is z_size
            nn.ConvTranspose2d(params.latent_dim, params.fBase * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(params.fBase * 4),
            nn.ReLU(inplace=True),
            # state size: (ngf * 8) x 4 x 4
            nn.ConvTranspose2d(params.fBase * 4, params.fBase * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(params.fBase * 2),
            nn.ReLU(inplace=True),
            # state size: (ngf * 4) x 8 x 8
            nn.ConvTranspose2d(params.fBase * 2, params.fBase, 4, 2, 1, bias=False),
            nn.BatchNorm2d(params.fBase),
            nn.ReLU(inplace=True),
            # state size: (ngf * 2) x 16 x 16
            nn.ConvTranspose2d(params.fBase, imgChans, 4, 2, 1, bias=False),
            # state size: out_size x 64 x 64
        )
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, z):
        pre_shape = z.size()[:-1]
        z = z.view(-1, z.size(-1), 1, 1)
        d = self.dec(z)
        d = d.view(*pre_shape, *torch.Size([1, 32, 32]))
        return d, torch.tensor(0.1).to(z.device)  # or 0.05


class RotatedMNIST(VAE):
    """ Derive a specific sub-class of a VAE for RotatedMNIST. """
    def __init__(self, params):
        super(RotatedMNIST, self).__init__(
            Sparse,                  # prior
            eval(params.posterior),  # posterior
            Laplace,            # likelihood
            Enc(params),
            Dec(params),
            params
        )
        self.modelName = 'rotatedmnist'

    @property
    def pz_params(self):
        return self.gamma.mul(1), self._pz_mu.mul(1), \
            torch.sqrt(self.prior_variance_scale * self._pz_logvar.size(-1) * F.softmax(self._pz_logvar, dim=1))

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device="cuda"):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        tx = transforms.Compose([
            transforms.Resize(size=(32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda p: p.clamp(Constants.eta, 1 - Constants.eta)),
            transforms.RandomRotation(180)
        ])
        train_loader = DataLoader(
            datasets.MNIST('../data/rotated_mnist', train=True, download=True, transform=tx),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
        test_loader = DataLoader(
            datasets.MNIST('../data/rotated_mnist', train=False, download=True, transform=tx),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
        return train_loader, test_loader

    def generate(self, runPath, epoch):
        N, K = 64, 9
        _, means, samples = super(RotatedMNIST, self).generate(N, K)
        save_image(means.data.cpu(), '{}/gen_means_{:03d}.png'.format(runPath, epoch))
        # wrangle things so they come out tiled
        samples = samples.view(K, N, *samples.size()[1:]).transpose(0, 1)
        s = [make_grid(t, nrow=int(math.sqrt(K)), padding=0) for t in samples.data.cpu()]
        save_image(torch.stack(s),
                   '{}/gen_samples_{:03d}.png'.format(runPath, epoch),
                   nrow=int(math.sqrt(N)))

    def reconstruct(self, data, runPath, epoch):
        recon = super(RotatedMNIST, self).reconstruct(data[:8])
        comp = torch.cat([data[:8], recon])
        save_image(comp.data.cpu(), '{}/recon_{:03d}.png'.format(runPath, epoch))
