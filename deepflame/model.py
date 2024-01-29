import torch
from torch import nn

from deepflame.trainer import Trainer
from deepflame.utils import normalize, denormalize, boxcox, inv_boxcox

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


class DFNN(Trainer): # It is possible to use nn.Module as the base class
    def __init__(
        self,
        n_species: int = 41,
        enc_size: list[int] = [256, 512],
        dec_size: list[int] = [512, 256],
    ):  # TODO: add inert gas index
        super().__init__()
        self.example_input_array = torch.zeros(1, 2 + n_species).split(
            [1, 1, n_species], dim=1
        )

        enc = MLP(
            [
                2 + n_species,
                *enc_size,
            ]
        )
        dec = nn.ModuleList(
            [MLP([enc_size[-1], *dec_size, 1]) for _ in range(n_species)]
        )
        dec.forward = lambda x: torch.stack([m(x) for m in dec], dim=2).squeeze()
        self.model = nn.Sequential(
            enc, dec
        )  # Forward: cat(T,P,Y_norm) of shape [batch, 2+ns] -> Y_delta[batch, ns]

        # # The old ways
        # layers = [2 + n_species, 400, 200, 100, 1]
        # self.model = nn.Modulelist([MLP(layers) for _ in range(n_species)])
        # self.model.forward = lambda x: torch.stack([m(x) for m in self.model], dim=2).squeeze()

        print(self.model)
        # torch.compile(self.model) # TODO: add config on torch.compile
        self.save_hyperparameters() # available if using LightningModule as base class

    # @torch.compile()
    def forward(self, T_in, P_in, Y_in):
        P_in /= 101325.
        # T_in -= self.model.T # TODO: normalize T
        Y_in_t = boxcox(Y_in)
        Y_in_norm = normalize(
            Y_in_t,
            self.model.Y_in_t_mean,
            self.model.Y_in_t_std,
        )
        Y_pred_t_delta_norm = self.model(torch.cat([T_in, P_in, Y_in_norm], dim=1))

        Y_pred_t_delta = denormalize(
            Y_pred_t_delta_norm,
            self.model.Y_t_delta_mean,
            self.model.Y_t_delta_std,
        )
        Y_pred_t = Y_in_t + Y_pred_t_delta
        return Y_pred_t
        Y_pred = inv_boxcox(Y_pred_t)
        Y_pred[Y_pred < 0] = 0
        return Y_pred
