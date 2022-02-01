from torch import nn

class EditNTS(nn.Module):

    def __init__(self, config):
        super(EditNTS, self).__init__()

    def forward(self):
        print("Forward")
