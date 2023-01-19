import math

import torch
from torch import nn

import numpy as np

from config import Config


# pulled from Dr. Karpathy's minGPT implementation
class GELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


# Encoder block used to help create latent representation of pre-evolution pokemon
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, device=torch.device("cpu"), only_conv=False):
        super().__init__()

        self.only_conv = only_conv

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding="same", device=device)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.b_norm = nn.BatchNorm2d(out_channels, device=device)
        self.gelu = GELU()
    

    def forward(self, x):
        out = self.conv(x)

        if self.only_conv == False:
            out = self.pool(out)
            out = self.b_norm(out)

            return self.gelu(out)

        return out


# Decoder block used to help construct evolved pokemon from latent representation
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, device=torch.device("cpu"), no_upsample=False):
        super().__init__()

        self.no_upsample = no_upsample

        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding="same", device=device)
        self.b_norm = nn.BatchNorm2d(out_channels, device=device)
        self.gelu = GELU()
    

    def forward(self, x):
        out = x
        if self.no_upsample == False:
            out = self.upsample(out)

        out = self.conv(out)
        out = self.b_norm(out)

        return self.gelu(out)


# VAE extracts latent features from pre-evolution image to produce evolution image
class VAE(nn.Module):
    def __init__(self, config, device=torch.device("cpu")):
        super().__init__()

        self.device = device

        ### ENCODER - extracts latent features from pre-evolution images

        # input: batch_size x 3 x 128 x 128, output: batch_size x 32 x 64 x 64
        self.encoder_blocks = [EncoderBlock(3, config.encoder_in_channels, device)]
        # input: batch_size x 32 x 64 x 64, output: batch_size x 64 x 32 x 32
        self.encoder_blocks += [EncoderBlock(config.encoder_in_channels, config.encoder_in_channels*2, device)]
        # input: batch_size x 64 x 32 x 32, output: batch_size x 128 x 16 x 16
        self.encoder_blocks += [EncoderBlock(config.encoder_in_channels*2, config.encoder_in_channels*4, device)]
        # input: batch_size x 128 x 16 x 16, output: batch_size x 256 x 8 x 8
        self.encoder_blocks += [EncoderBlock(config.encoder_in_channels*4, config.encoder_in_channels*8, device)]
        # input: batch_size x 256 x 8 x 8, output: batch_size x 512 x 4 x 4
        self.encoder_blocks += [EncoderBlock(config.encoder_in_channels*8, config.encoder_in_channels*16, device)]

        # input: batch_size x 512 x 4 x 4, output: batch_size x 1024 x 4 x 4
        self.mean_block = EncoderBlock(config.encoder_in_channels*16, config.encoder_in_channels*32, device, True)
        # input: batch_size x 512 x 4 x 4, output: batch_size x 1024 x 4 x 4
        self.log_var_block = EncoderBlock(config.encoder_in_channels*16, config.encoder_in_channels*32, device, True)

        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)


        ### DECODER - generates evolution images from latent pre-evolution latent representation

        # input: batch_size x 1024 x 4 x 4, output: batch_size x 512 x 4 x 4
        self.decoder_blocks = [DecoderBlock(int(config.decoder_in_channels), int(config.decoder_in_channels/2), device, True)]
        # input: batch_size x 512 x 4 x 4, output: batch_size x 256 x 8 x 8
        self.decoder_blocks += [DecoderBlock(int(config.decoder_in_channels/2), int(config.decoder_in_channels/4), device)]
        # input: batch_size x 256 x 8 x 8, output: batch_size x 128 x 16 x 16
        self.decoder_blocks += [DecoderBlock(int(config.decoder_in_channels/4), int(config.decoder_in_channels/8), device)]
        # input: batch_size x 128 x 16 x 16, output: batch_size x 64 x 32 x 32
        self.decoder_blocks += [DecoderBlock(int(config.decoder_in_channels/8), int(config.decoder_in_channels/16), device)]
        # input: batch_size x 64 x 32 x 32, output: batch_size x 64 x 32 x 32
        self.decoder_blocks += [DecoderBlock(int(config.decoder_in_channels/16), int(config.decoder_in_channels/16), device)]

        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)

        # input: batch_size x 64 x 32 x 32, output: batch_size x 3 x 128 x 128
        self.final_upsample = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(int(config.decoder_in_channels/16), 3, kernel_size=(3, 3), padding="same", device=device)

        # output needs to be normalized between -1->1
        self.tanh = nn.Tanh()

    
    def reparameterization(self, mean, variance):
        epsilon = torch.randn_like(variance).to(self.device)        
        z = mean + variance * epsilon
        return z

    
    def forward(self, x):
        out = x
        residuals = []
        for i in range(len(self.encoder_blocks)):
            out = self.encoder_blocks[i](out)
            if i > 1 and i < len(self.encoder_blocks):
                residuals += [out]

        mean = self.mean_block(out)
        log_var = self.log_var_block(out)

        out = self.reparameterization(mean, torch.exp(0.5 * log_var))

        # residual connections to feed info about pre-evolved pokemon to synthesize better output
        for i in range(len(self.decoder_blocks)):
            if i > 0 and i < len(residuals)+1:
                out += residuals[len(residuals)-i]
            out = self.decoder_blocks[i](out)

        out = self.final_upsample(out)
        out = self.conv1(out)
        
        return self.tanh(out), mean, log_var