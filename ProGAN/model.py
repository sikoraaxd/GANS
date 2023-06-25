import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2


factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]


class WeightsScaledConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super(WeightsScaledConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = gain / (in_channels * (kernel_size ** 2)) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    
    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)



class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True)) + self.epsilon



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pixelnorm=True):
        super(ConvBlock, self).__init__()
        self.conv1 = WeightsScaledConv(in_channels, out_channels)
        self.conv2 = WeightsScaledConv(out_channels, out_channels)
        self.lrelu = nn.LeakyReLU(0.2)
        self.pixel_norm = PixelNorm()
        self.use_pixelnorm = use_pixelnorm


    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.pixel_norm(x) if self.use_pixelnorm else x
        x = self.lrelu(self.conv2(x))
        x = self.pixel_norm(x) if self.use_pixelnorm else x
        return x



class Generator(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels=3):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim, in_channels, 4, 1, 0),
            nn.LeakyReLU(0.2),
            WeightsScaledConv(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm(),
        )

        self.initial_rgb = WeightsScaledConv(in_channels, img_channels, kernel_size=1, stride=1, padding=0)
        self.progressive_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([self.initial_rgb])

        for i in range(len(factors)-1):
            conv_in_channels = int(in_channels * factors[i])
            conv_out_channels = int(in_channels * factors[i+1])
            self.progressive_blocks.append(ConvBlock(conv_in_channels, conv_out_channels))
            self.rgb_layers.append(WeightsScaledConv(conv_out_channels, img_channels, kernel_size=1, stride=1, padding=0))
            

    def fade_in(self, alpha, upscaled, generated):
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)


    def forward(self, x, alpha, steps):
        out = self.initial(x)

        if steps == 0:
            return self.initial_rgb(out)
        
        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode="nearest")
            out = self.progressive_blocks[step](upscaled)

        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out)

        return self.fade_in(alpha, final_upscaled, final_out)



class Discriminator(nn.Module):
    def __init__(self, in_channels, img_channels=3):
        super(Discriminator, self).__init__()
        self.progressive_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])
        self.lrelu = nn.LeakyReLU(0.2)
        
        for i in range(len(factors) - 1, 0, -1):
            conv_in_channels = int(in_channels * factors[i])
            conv_out_channels = int(in_channels * factors[i-1])
            self.progressive_blocks.append(ConvBlock(conv_in_channels, conv_out_channels, use_pixelnorm=False))
            self.rgb_layers.append(WeightsScaledConv(img_channels, conv_in_channels, kernel_size=1, stride=1, padding=0))

        self.initial_rgb = WeightsScaledConv(img_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.rgb_layers.append(self.initial_rgb)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.final_block = nn.Sequential(
            WeightsScaledConv(in_channels+1, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            WeightsScaledConv(in_channels, in_channels, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            WeightsScaledConv(in_channels, 1, kernel_size=1, stride=1, padding=0)
        )


    def fade_in(self, alpha, downscaled, out):
        return alpha * out + (1 - alpha) * downscaled
    

    def minibatch_std(self, x):
        batch_statistics = (
            torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        )
        return torch.cat([x, batch_statistics], dim=1)
    
    
    def forward(self, x, alpha, steps):
        current_step = len(self.progressive_blocks) - steps
        out = self.lrelu(self.rgb_layers[current_step](x))

        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)
        
        downscaled = self.lrelu(self.rgb_layers[current_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.progressive_blocks[current_step](out))
        out = self.fade_in(alpha, downscaled, out)

        for step in range(current_step + 1, len(self.progressive_blocks)):
            out = self.progressive_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)
