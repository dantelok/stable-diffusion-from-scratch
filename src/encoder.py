import torch
from torch import nn
from torch.nn import functional as F
from src.decoder import VAEAttentionBlock, VAEResidualBlock


class VAEEncoder(nn.Sequential):
    # Reducing size but increasing channels over time
    def __init__(self):
        super().__init__(
            # (Batch_Size, Channel, Height, Width) -> (Batch_Size, 128, Height, Width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAEResidualBlock(128, 128),

            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAEResidualBlock(128, 128),

            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height/2 , Width/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # (Batch_Size, 256, Height/2 , Width/2) -> (Batch_Size, 256, Height/2 , Width/2)
            VAEResidualBlock(128, 256),

            # (Batch_Size, 256, Height/2 , Width/2) -> (Batch_Size, 256, Height/2 , Width/2)
            VAEResidualBlock(256, 256),

            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height/2 , Width/2)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # (Batch_Size, 256, Height/2 , Width/2) -> (Batch_Size, 256, Height/4 , Width/4)
            VAEResidualBlock(256, 512),

            # (Batch_Size, 256, Height/4 , Width/4) -> (Batch_Size, 256, Height/4 , Width/4)
            VAEResidualBlock(512, 512),

            # (Batch_Size, 256, Height/4, Width/4) -> (Batch_Size, 256, Height/8 , Width/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            # (Batch_Size, 512, Height/8 , Width/8) -> (Batch_Size, 512, Height/8 , Width/8)
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEAttentionBlock(512),
            VAEResidualBlock(512, 512),

            nn.GroupNorm(32, 512),

            nn.SiLU(),

            # (Batch_Size, 512, Height/8 , Width/8) -> (Batch_Size, 8, Height/8 , Width/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # (Batch_Size, 8, Height/8 , Width/8) -> (Batch_Size, 8, Height/8 , Width/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        :param x: (Batch_Size, Channel, Height, Width)
        :param noise: (Batch_Size, Out_Channel, Height/8, Width/9)
        :return:
        """

        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        # (Batch_Size, 8, Height/8 , Width/8) -> 2 Tensors of shape (Batch_Size, 4, Height/8 , Width/8)
        mean, log_variance = torch.chunk(x, chunks=2, dim=1)

        # (Batch_Size, 4, Height/8 , Width/8) -> (Batch_Size, 4, Height/8 , Width/8)
        log_variance = torch.clamp(log_variance, min=-30, max=20)

        variance = torch.exp(log_variance)

        std = torch.sqrt(variance)

        # N(0, 1) -> N(mean, variance)?
        x = mean + std * noise

        # Scale the output by a constant
        x *= 0.18215

        return x
