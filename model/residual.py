import torch
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, sublayer):
        super().__init__()
        self.sublayer = sublayer
        
    def forward(self, input):
        sublayer_output = self.sublayer(input)
        return input + sublayer_output 
