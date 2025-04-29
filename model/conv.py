import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__ (self,
                  in_channels,
                  out_channels,
                  kernel_size = 3,
                  stride = 1,
                  padding = "same",
                  ):
        super().__init__()

        if out_channels >= 32:
            group =  out_channels // 32 
        else:
            group = 1 

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm1 = nn.GroupNorm(group, out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.norm2 = nn.GroupNorm(group,out_channels)


    #(batch, in_channel, width, width) -> (batch, out_channel, width, width)
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x
        


