import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.layer = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        return self.layer(x)