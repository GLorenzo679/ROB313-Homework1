import torch.nn as nn
from models.decoder import Decoder
from models.encoder import Encoder
from models.quantizer import VectorQuantizer


class VQVAE(nn.Module):
    def __init__(self, latent_dim=64, num_embeddings=512, embedding_dim=64):
        super(VQVAE, self).__init__()

        self.encoder = Encoder(latent_dim)
        self.pre_quantization_conv = nn.Conv2d(
            latent_dim, embedding_dim, kernel_size=1, stride=1
        )
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, 0.25)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        z = self.pre_quantization_conv(z)
        loss, quantized, perplexity, _ = self.quantizer(z)
        x_recon = self.decoder(quantized)

        return x_recon, loss, perplexity
