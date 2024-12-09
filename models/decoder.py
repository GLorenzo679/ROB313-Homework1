import torch.nn as nn
from models.residual import ResidualStack


class Decoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            ResidualStack(
                latent_dim, latent_dim, latent_dim, 2
            ),  # Output: [latent_dim, 7, 7]
            nn.ConvTranspose2d(
                latent_dim, 64, kernel_size=3, stride=1, padding=1
            ),  # Output: [64, 7, 7]
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1
            ),  # Output: [32, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 1, kernel_size=4, stride=2, padding=1
            ),  # Output: [1, 28, 28]
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(x)
