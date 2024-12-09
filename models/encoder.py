import torch.nn as nn
from models.residual import ResidualStack


class Encoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                1, 32, kernel_size=4, stride=2, padding=1
            ),  # Output: [32, 14, 14]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: [64, 7, 7]
            nn.ReLU(),
            nn.Conv2d(
                64, latent_dim, kernel_size=3, stride=1, padding=1
            ),  # Output: [latent_dim, 7, 7]
            ResidualStack(
                latent_dim, latent_dim, latent_dim, 2
            ),  # Output: [latent_dim, 7, 7]
        )

    def forward(self, x):
        return self.encoder(x)
