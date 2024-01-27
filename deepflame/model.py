from torch import nn

class MLP(nn.Sequential):
    # https://pytorch.org/rl/_modules/torchrl/modules/models/models.html#MLP
    def __init__(self, layer_sizes, activation=nn.GELU):
        layers = []
        depth = len(layer_sizes) - 1
        for i in range(depth):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i != depth - 1:
                layers.append(activation())
        super().__init__(*layers)
