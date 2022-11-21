import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
from einops import rearrange

import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    positional encoding - 1 x max_len x dim_size
    """
    def __init__(self, d_model: int, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x = batch, seq, dim
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)

class MusicTaggingTransformer(nn.Module):
    def __init__(
        self,
        conv_ndim=16,
        n_mels=128,
        sample_rate=22050,
        n_fft=1024,
        f_min=0,
        f_max=11025,
        attention_ndim=256,
        attention_nheads=8,
        attention_nlayers=4,
        attention_max_len=512,
        dropout=0.1,
        n_seq_cls=1,
        n_token_cls=1,
    ):
        super(MusicTaggingTransformer, self).__init__()
        # Input preprocessing
        self.spec = self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                                     n_fft=n_fft,
                                                                     f_min=f_min,
                                                                     f_max=f_max,
                                                                     n_mels=n_mels,
                                                                     power=2)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # Input embedding
        self.frontend = ResFrontEnd(conv_ndim, attention_ndim, n_mels)

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, attention_max_len + 1, attention_ndim))
        self.cls_token = nn.Parameter(torch.randn(attention_ndim))

        # transformer
        self.transformer = Transformer(
            attention_ndim,
            attention_nlayers,
            attention_nheads,
            attention_ndim // attention_nheads,
            attention_ndim * 4,
            dropout,
        )
        self.to_latent = nn.Identity()
        self.dropout = nn.Dropout(dropout)

        # projection for sequence classification
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(attention_ndim), nn.Linear(attention_ndim, n_seq_cls)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (batch, time)
        Returns:
            x (torch.Tensor): (batch, n_seq_cls)
        """
        # Input preprocessing
        x = self.spec(x)
        x = self.amplitude_to_db(x)
        x = x.unsqueeze(1)

        # Input embedding
        x = self.frontend(x)

        # Positional embedding with a [CLS] token
        cls_token = self.cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.pos_embedding[:, : x.size(1)]
        x = self.dropout(x)

        # transformer
        x = self.transformer(x)

        # projection for sequence classification
        x = self.to_latent(x[:, 0])
        x = self.mlp_head(x)
        x = self.sigmoid(x)
        return x

class Res2DMaxPoolModule(nn.Module):
    def __init__(self, input_channels, output_channels, pooling=2):
        super(Res2DMaxPoolModule, self).__init__()
        self.conv_1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(output_channels)
        self.conv_2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)

        # residual
        self.diff = False
        if input_channels != output_channels:
            self.conv_3 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
            self.bn_3 = nn.BatchNorm2d(output_channels)
            self.diff = True

    def forward(self, x):
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))
        if self.diff:
            x = self.bn_3(self.conv_3(x))
        out = x + out
        out = self.mp(self.relu(out))
        return out


class ResFrontEnd(nn.Module):
    """
    Evaluation of CNN based Music Tagging.
    Won et al., 2020
    
    Note that, different from the original work, we only stack 3 convolutional layers instead of 7.
    After the convolution layers, we flatten the time-frequency representation to be a vector.
    """

    def __init__(self, conv_ndim, attention_ndim, nfreq, nharmonics=1):
        super(ResFrontEnd, self).__init__()
        self.input_bn = nn.BatchNorm2d(nharmonics)
        self.layer1 = Res2DMaxPoolModule(nharmonics, conv_ndim, pooling=(2, 2))
        self.layer2 = Res2DMaxPoolModule(conv_ndim, conv_ndim, pooling=(2, 2))
        self.layer3 = Res2DMaxPoolModule(conv_ndim, conv_ndim, pooling=(2, 1))
        fc_ndim = nfreq // 2 // 2 // 2 * conv_ndim
        self.fc = nn.Linear(fc_ndim, attention_ndim)

    def forward(self, hcqt):
        # batch normalization
        out = self.input_bn(hcqt)

        # CNN
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        # permute and channel control
        b, c, f, t = out.shape
        out = out.permute(0, 3, 1, 2)  # batch, time, conv_ndim, freq
        out = out.contiguous().view(b, t, -1)  # batch, time, fc_ndim
        out = self.fc(out)  # batch, time, attention_ndim
        return out


# Transformer modules
"""
    Referenced PyTorch implementation of Vision Transformer by Lucidrains.
    https://github.com/lucidrains/vit-pytorch.git
"""
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            # mask = F.pad(mask.flatten(1), (1, 0), value=True) # I already pad cls token
            mask = mask.type(torch.bool)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(
                            PreNorm(
                                dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
                            )
                        ),
                        Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))),
                    ]
                )
            )

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class Res_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=2):
        super(Res_2d, self).__init__()
        # convolution
        self.conv_1 = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
        self.bn_1 = nn.BatchNorm2d(output_channels)
        self.conv_2 = nn.Conv2d(output_channels, output_channels, shape, padding=shape//2)
        self.bn_2 = nn.BatchNorm2d(output_channels)

        # residual
        self.diff = False
        if (stride != 1) or (input_channels != output_channels):
            self.conv_3 = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
            self.bn_3 = nn.BatchNorm2d(output_channels)
            self.diff = True
        self.relu = nn.ReLU()

    def forward(self, x):
        # convolution
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))

        # residual
        if self.diff:
            x = self.bn_3(self.conv_3(x))
        out = x + out
        out = self.relu(out)
        return out