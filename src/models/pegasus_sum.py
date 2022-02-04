from torch import nn

TRAIN_ORIGINAL_DATA_PATH = "resources/wikismall/PWKP_108016.tag.80.aner.ori.train.src"
TRAIN_SIMPLE_DATA_PATH = "resources/wikismall/PWKP_108016.tag.80.aner.ori.train.dst"


class PegasusSumModel(nn.Module):

    def __init__(self, config):
        super(PegasusSumModel, self).__init__()

    def forward(self):
        print("Forward")