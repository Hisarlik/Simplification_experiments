import torch
from torch import nn




class EditNTSModel(nn.Module):

    def __init__(self, config):
        super(EditNTSModel, self).__init__()

    def forward(self):
        print("Forward")