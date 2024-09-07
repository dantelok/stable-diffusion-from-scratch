import torch
from torch import nn
from torch.nn import functional as F
from src.attention import SelfAttention


class VAEAttentionBlock(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        # Group norm = Layer norm (Group of data rows) + Batch norm (Group of features)
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: (Batch_size, Features, Height, Width)
        :return:
        """

        residue = x
        n, c, h, w = x.shape

        # (Batch_size, Features, Height, Width) -> (Batch_size, Features, Height * Width)
        x = x.view(n, c, h*w)

        # (Batch_size, Features, Height * Width) -> (Batch_size, Height * Width, Features)
        x = x.transpose(-1, -2)

        # (Batch_size, Height * Width, Features) -> (Batch_size, Height * Width, Features)
        x = self.attention(x)

        # (Batch_size, Height * Width, Features) -> (Batch_size, Features, Height * Width)
        x = x.transpose(-1, -2)

        # (Batch_size, Features, Height * Width) -> (Batch_size, Features, Height, Width)
        x = x.view(n, c, h, w)

        x += residue

        return x


class VAEResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (Batch_size, In_Channels, Height, Width)
        :return:
        """
        residual = x
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residual)


class VAEDecoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            VAEResidualBlock(512, 512),
            VAEAttentionBlock(512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),

            # (Batch_size, 512, Height/8, Width/8) -> (Batch_size, 512, Height/4, Width/4)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),

            # (Batch_size, 512, Height/4, Width/4) -> (Batch_size, 512, Height/2, Width/2)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAEResidualBlock(512, 256),
            VAEResidualBlock(256, 256),
            VAEResidualBlock(256, 256),

            # (Batch_size, 256, Height/2, Width/2) -> (Batch_size, 256, Height, Width)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            VAEResidualBlock(256, 128),
            VAEResidualBlock(128, 128),
            VAEResidualBlock(128, 128),

            nn.GroupNorm(32, 128),

            nn.SiLU(),

            # (Batch_size, 128, Height, Width) -> (Batch_size, 3, Height, Width)
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (Batch_size, 4, Height/8, Width/8)
        :return:
        """

        x /= 0.18215

        for module in self:
            x = module(x)

        # (Batch_size, 3, Height, Width)
        return x
