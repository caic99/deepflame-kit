import torch
from torch import nn
from typing import List
from deepflame.trainer import DFTrainer
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

# To infer without Lightning,
# i. use `nn.Module` instead of `DFTrainer` as the base class for `DFNN` below, and
# ii. remove `from deepflame.trainer import DFTrainer` of this file.
class DFNN(DFTrainer):
    def __init__(
        self,
        n_species: int = 41,
        inert_index: int = -1,
        lmbda: float = 0.1,
        layer_sizes: List[int] = [512, 256, 128],
    ):
        super().__init__()
        self.example_input_array = torch.zeros(1, 2 + n_species).split(
            [1, 1, n_species], dim=1
        )
        layers = [2 + n_species, *layer_sizes, 1]
        self.model = nn.ModuleList([MLP(layers) for _ in range(n_species)])
        self.model.forward = lambda x: torch.cat([m(x) for m in self.model], dim=1)
        self.model.inert_index = inert_index
        self.model.lmbda = lmbda
        print(self.model)
        # model = torch.compile(self.model) # TODO: add config on torch.compile
        try:
            self.save_hyperparameters()
        except AttributeError: # is only an `nn.Module`
            pass

    # @torch.compile()
    def forward(self, T_in, P_in, Y_t_in):
        T_norm_in = normalize(T_in, self.model.T_in_mean, self.model.T_in_std)
        P_norm_in = normalize(P_in, self.model.P_in_mean, self.model.P_in_std)
        P_norm_in[...] = 0 # HACK: fix P_in to 0
        Y_n_in = normalize(
            Y_t_in,
            self.model.Y_t_in_mean,
            self.model.Y_t_in_std,
        )
        Y_dt_n_pred = self.model(torch.cat([T_norm_in, P_norm_in, Y_n_in], dim=1))
        Y_dt_pred = denormalize(
            Y_dt_n_pred,
            self.model.Y_dt_mean,
            self.model.Y_dt_std,
        )
        Y_dt_pred[:, self.model.inert_index] = 0
        Y_t_pred = Y_t_in + Y_dt_pred
        Y_pred = inv_boxcox(Y_t_pred, self.model.lmbda)
        # Y_pred[Y_pred < 0] = 0
        return Y_pred, Y_dt_n_pred

    def predict(self, T_in, P_in, Y_in):
        """Interface for Infer.
        Note: Y_in contains inert gas.
        """
        # Don't forget to set `with torch.no_grad()` when calling this function
        Y_t_in = boxcox(Y_in, self.model.lmbda)
        Y_pred, Y_dt_n_pred = self.forward(T_in, P_in, Y_t_in)
        # Normalize ΣY = 1 while maintaining the inert gas fraction
        factor = (1 - Y_pred[:, self.model.inert_index]) / (
            Y_pred.sum(dim=1) - Y_pred[:, self.model.inert_index]
        )
        Y_pred *= factor.unsqueeze(dim=-1)
        Y_pred[:, self.model.inert_index] /= factor  # keep inert unchanged
        return Y_pred - Y_in
