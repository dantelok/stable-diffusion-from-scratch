import torch
from torch import nn
from torch.nn import functional as F
from src.attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):

    def __init__(self, n_embed: int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed, 4 * n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (1, 320)
        :return:
        """

        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)

        return x


class UNETResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, num_channels=in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, num_channels=out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, feature: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        residue = feature
        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)

        feature = self.conv_feature(feature)
        time = F.silu(time)
        time = self.linear_time(time)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)
        merged = merged + self.residual_layer(residue)

        return merged


class UNETAttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embed: int, d_context=768):
        super().__init__()
        channels = n_head * n_embed
        self.groupnorm = nn.GroupNorm(32, num_channels=channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4*channels*2)
        self.linear_geglu_2 = nn.Linear(4*channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        :param x: (Batch_size, Features, Height, Width)
        :param context: (Batch_size, Seq_Len, Dim)
        :return:
        """
        residue_long = x

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x = self.groupnorm(x)
        x = self.conv_input(x)
        n, c, h, w = x.shape

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x = x.view((n, c, h * w))

        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features)
        x = x.transpose(-1, -2)

        # Normalization + Self-Attention with skip connection
        # (Batch_Size, Height * Width, Features)
        residue_short = x

        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short
        residue_short = x

        # Normalization + Cross-Attention with skip connection
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)
        x += residue_short
        residue_short = x

        # Normalization + FFN with GeGLU and skip connection
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_3(x)

        # GeGLU as implemented in the original code: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/attention.py#L37C10-L37C10
        # (Batch_Size, Height * Width, Features) -> two tensors of shape (Batch_Size, Height * Width, Features * 4)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)

        # Element-wise product: (Batch_Size, Height * Width, Features * 4) * (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features * 4)
        x = x * F.gelu(gate)

        # (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features)
        x = self.linear_geglu_2(x)
        x += residue_short

        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
        x = x.transpose(-1, -2)

        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        x = x.view((n, c, h, w))

        # Final skip connection between initial input and output of the block
        x = self.conv_output(x) + residue_long

        return x


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height*2, Width*2)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        return x


class SwitchSequential(nn.Sequential):

    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNETAttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNETResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            # (Batch_Size, 4, Height / 8, Width / 8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(UNETResidualBlock(320, 320), UNETAttentionBlock(8, 40)),
            SwitchSequential(UNETResidualBlock(320, 320), UNETAttentionBlock(8, 40)),

            # (Batch_Size, 320, Height / 8, Width / 8) -> # (Batch_Size, 320, Height / 16, Width / 16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNETResidualBlock(320, 640), UNETAttentionBlock(8, 80)),
            SwitchSequential(UNETResidualBlock(640, 640), UNETAttentionBlock(8, 80)),

            # (Batch_Size, 640, Height / 16, Width / 16) -> # (Batch_Size, 640, Height / 32, Width / 32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNETResidualBlock(640, 1280), UNETAttentionBlock(8, 160)),
            SwitchSequential(UNETResidualBlock(1280, 1280), UNETAttentionBlock(8, 160)),

            # (Batch_Size, 1280, Height / 32, Width / 32) -> # (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNETResidualBlock(1280, 1280)),

            # (Batch_Size, 1280, Height / 64, Width / 64) -> # (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNETResidualBlock(1280, 1280)),
        ])

        self.bottleneck = SwitchSequential(
            UNETResidualBlock(1280, 1280),
            UNETAttentionBlock(8, 160),
            UNETResidualBlock(1280, 1280),
        )

        self.decoders = nn.ModuleList([
            SwitchSequential(UNETResidualBlock(2560, 1280)),
            SwitchSequential(UNETResidualBlock(2560, 1280)),

            SwitchSequential(UNETResidualBlock(2560, 1280), Upsample(1280)),
            SwitchSequential(UNETResidualBlock(2560, 1280), UNETAttentionBlock(8, 160)),
            SwitchSequential(UNETResidualBlock(2560, 1280), UNETAttentionBlock(8, 160)),

            SwitchSequential(UNETResidualBlock(1920, 1280), UNETAttentionBlock(8, 160), Upsample(1280)),
            SwitchSequential(UNETResidualBlock(1920, 640), UNETAttentionBlock(8, 80)),
            SwitchSequential(UNETResidualBlock(1280, 640), UNETAttentionBlock(8, 80)),

            SwitchSequential(UNETResidualBlock(960, 640), UNETAttentionBlock(8, 80), Upsample(640)),
            SwitchSequential(UNETResidualBlock(960, 320), UNETAttentionBlock(8, 40)),
            SwitchSequential(UNETResidualBlock(640, 320), UNETAttentionBlock(8, 40)),
            SwitchSequential(UNETResidualBlock(640, 320), UNETAttentionBlock(8, 40)),
        ])

    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layers(x, context, time)

        return x


class UNETOutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (Batch_size, 320, Height/8, Width/8)
        :return: (Batch_size, 4, Height/8, Width/8)
        """

        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)

        return x


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNETOutputLayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        :param latent: (Batch_size, 4, Height/8, Width/8)
        :param context: (Batch_size, Seq_Len, Dim)
        :param time: (1, 320)
        :return: (Batch, 4, Height / 8, Width / 8)
        """
        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        # (Batch, 4, Height / 8, Width / 8) -> (Batch, 320, Height / 8, Width / 8)
        output = self.unet(latent, context, time)

        # (Batch, 320, Height / 8, Width / 8) -> (Batch, 4, Height / 8, Width / 8)
        output = self.final(output)

        return output
