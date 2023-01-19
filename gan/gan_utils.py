import math

import torch
from torch import nn

from gan.gan_config import Config


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
    def __init__(self, in_channels, out_channels, device=torch.device("cpu")):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding="same", device=device)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.b_norm = nn.BatchNorm2d(out_channels, device=device)
        self.gelu = GELU()
    

    def forward(self, x):
        out = self.conv(x)
        out = self.pool(out)
        out = self.b_norm(out)

        return self.gelu(out)


# Decoder block used to help construct evolved pokemon from latent representation
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, device=torch.device("cpu")):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding="same", device=device)
        self.b_norm = nn.BatchNorm2d(out_channels, device=device)
        self.gelu = GELU()
    

    def forward(self, x):
        out = self.upsample(x)
        out = self.conv(out)
        out = self.b_norm(out)

        return self.gelu(out)


# Generator extracts latent features from pre-evolution image to produce evolution image
class Generator(nn.Module):
    def __init__(self, config, device=torch.device("cpu")):
        super().__init__()

        ### ENCODER - extracts latent features from pre-evolution images

        # input: batch_size x 3 x 128 x 128, output: batch_size x 64 x 64 x 64
        self.encoder_blocks = [EncoderBlock(3, config.encoder_in_channels, device)]
        # input: batch_size x 64 x 64 x 64, output: batch_size x 128 x 32 x 32
        self.encoder_blocks += [EncoderBlock(config.encoder_in_channels, config.encoder_in_channels*2, device)]
        # input: batch_size x 128 x 32 x 32, output: batch_size x 256 x 16 x 16
        self.encoder_blocks += [EncoderBlock(config.encoder_in_channels*2, config.encoder_in_channels*4, device)]

        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)


        ### DECODER - generates evolution images from latent pre-evolution latent representation

        # input: batch_size x 256 x 16 x 16, output: batch_size x 128 x 32 x 32
        self.decoder_blocks = [DecoderBlock(config.decoder_in_channels, int(config.decoder_in_channels/2), device)]
        # input: batch_size x 128 x 32 x 32, output: batch_size x 64 x 64 x 64
        self.decoder_blocks += [DecoderBlock(int(config.decoder_in_channels/2), int(config.decoder_in_channels/4), device)]

        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)

        # input: batch_size x 64 x 64 x 64, output: batch_size x 3 x 128 x 128
        self.final_upsample = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(int(config.decoder_in_channels/4), 3, kernel_size=(3, 3), padding="same", device=device)

        # output needs to be normalized between -1->1
        self.tanh = nn.Tanh()

    
    def forward(self, x):
        out = x
        for i in range(len(self.encoder_blocks)):
            out = self.encoder_blocks[i](out)

        # residual connections to feed info about pre-evolved pokemon to synthesize better output
        for i in range(len(self.decoder_blocks)):
            out = self.decoder_blocks[i](out)

        out = self.final_upsample(out)
        out = self.conv1(out)
        
        return self.tanh(out)


# Discriminator tries to identify which samples are made by the Generator vs. from the original dataset
class Discriminator(nn.Module):
    def __init__(self, config, device=torch.device("cpu")):
        super().__init__()

        ### ENCODER - extracts latent features from pre-evolution images

        # input: batch_size x 3 x 128 x 128, output: batch_size x 64 x 64 x 64
        self.encoder_blocks = [EncoderBlock(3, config.discriminator_in_channels, device)]
        # input: batch_size x 64 x 64 x 64, output: batch_size x 128 x 32 x 32
        self.encoder_blocks += [EncoderBlock(config.discriminator_in_channels, config.discriminator_in_channels*2, device)]
        # input: batch_size x 128 x 32 x 32, output: batch_size x 256 x 16 x 16
        self.encoder_blocks += [EncoderBlock(config.discriminator_in_channels*2, config.discriminator_in_channels*4, device)]
        # input: batch_size x 256 x 16 x 16, output: batch_size x 512 x 8 x 8
        self.encoder_blocks += [EncoderBlock(config.discriminator_in_channels*4, config.discriminator_in_channels*8, device)]

        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)

        self.flatten = nn.Flatten()
        self.final_linear = nn.Linear(config.discriminator_in_channels*8*8*8, 1)
        
        # output needs to be normalized between 0->1
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, x):
        out = x
        for i in range(len(self.encoder_blocks)):
            out = self.encoder_blocks[i](out)
        
        out = self.flatten(out)
        out = self.final_linear(out)
        
        return self.sigmoid(out)