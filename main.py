from pathlib import Path
from typing import List
import lightning as L
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

from lightning.pytorch.cli import LightningCLI
import yaml
from IPython import embed


def boxcox(
    x, lmbda=0.05
):  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.boxcox.html
    return (x**lmbda - 1) / lmbda  # if lmbda != 0 else log(x)


def inv_boxcox(y, lmbda=0.05):
    return (y * lmbda + 1) ** (1 / lmbda)  # if lmbda != 0 else exp(x)


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


class DFNN(L.LightningModule):
    def __init__(
        self,
        n_species: int = 41,
        enc_size: List[int] = [256, 512],
        dec_size: List[int] = [512, 256],
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
        # self.model = nn.ModuleList([MLP(layers) for _ in range(n_species)])
        # self.model.forward = lambda x: torch.stack([m(x) for m in self.model], dim=2).squeeze()

        print(self.model)
        self.save_hyperparameters()

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
    #     lr_scheduler = torch.optim.lr_scheduler.StepLR(
    #         optimizer, step_size=1, gamma=0.8
    #     )
    #     return [optimizer], [lr_scheduler]

    def setup(self, stage):
        # Transfer metadata to device
        self.dataset: DFDataSet = self.trainer.datamodule.dataset  # type: ignore
        for i in [
            "formation_enthalpies",
            "Y_in_t_mean",
            "Y_in_t_std",
            "Y_gt_t_mean",
            "Y_gt_t_std",
            "Y_t_delta_mean",
            "Y_t_delta_std",
            "T",
            "P",
            "time_step",
            "lmbda",
        ]:
            self.model.register_buffer(
                i,
                torch.tensor(
                    getattr(self.dataset, i),
                    dtype=torch.get_default_dtype(),
                    device=self.device,
                ),
            )
        # self.model = torch.compile(self.model)

        # TODO: test pipeline

    @torch.compile()
    def forward(self, T_in, P_in, Y_in):
        # see `DFDataset.__init__()` for normalization method.
        # Should take raw data as input for prediction;
        # TODO: move the use of self.dataset to self.model
        P_in -= self.model.P
        # T_in -= self.model.T # TODO: normalize T
        Y_in_t = self.dataset.transform(Y_in)
        Y_in_norm = self.dataset.norm(
            Y_in_t,
            self.model.Y_in_t_mean,
            self.model.Y_in_t_std,
        )
        Y_pred_t_delta_norm = self.model(torch.cat([T_in, P_in, Y_in_norm], dim=1))

        Y_pred_t_delta = self.dataset.denorm(
            Y_pred_t_delta_norm,
            self.model.Y_t_delta_mean,
            self.model.Y_t_delta_std,
        )
        Y_pred_t = Y_in_t + Y_pred_t_delta
        return Y_pred_t
        Y_pred = self.dataset.inv_transform(Y_pred_t)
        Y_pred[Y_pred < 0] = 0
        return Y_pred

    def training_step(self, batch, batch_idx):
        (
            T_in,
            P_in,
            Y_in,
            H_in,
            T_gt,
            P_gt,
            Y_gt,
            H_gt,
        ) = batch

        Y_pred_t = self.forward(T_in, P_in, Y_in)
        criterion = nn.L1Loss()

        Y_gt_t = self.dataset.transform(Y_gt)
        loss1 = criterion(Y_pred_t, Y_gt_t)

        Y_pred = self.dataset.inv_transform(Y_pred_t)
        Y_pred_sum = Y_pred.sum(axis=1)
        loss2 = criterion(
            Y_pred_sum,
            #   Y_gt.sum(axis=1))
            torch.ones_like(Y_pred_sum),
        )

        scale = (
            self.model.time_step * 1e13
        )  # prevent overflow introduced by large H and small time_step

        loss3 = (
            criterion((H_gt * Y_pred).sum(axis=1), (H_in * Y_in).sum(axis=1)) / scale
        )

        fe = self.model.formation_enthalpies
        loss4 = (
            criterion(
                (fe * Y_pred).sum(axis=1),
                (fe * Y_gt).sum(axis=1),
            )
            / scale
        )
        # Target: 3e-3, 1e-6, 1e+8, 1e+8
        loss = loss1 + loss2 + loss3 + loss4
        # TODO: change weights
        # TODO: add more losses

        # print(f"{loss=:.4e}, {loss1=:.4e}, {loss2=:.4e}, {loss3=:.4e}, {loss4=:.4e}")
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            # logger=False,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log("val_loss", loss)
        self.log("learning_rate", self.optimizers().optimizer.param_groups[0]["lr"])  # type: ignore

    def test_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log("test_loss", loss)

    def predict_step(self, batch, batch_idx):
        pass  # TODO: predict loop


class DFDataSet(Dataset):
    def __init__(
        self,
        data_path: Path,
        config_path: Path,
        formation_enthalpies_path: Path,
        lmbda=0.05,
    ):
        super().__init__()
        # Check n_species
        self.config = yaml.load(open(config_path), Loader=yaml.FullLoader)
        assert len(self.config["phases"]) == 1, "Only single phase is supported"
        phase = self.config["phases"][0]
        self.n_species = len(phase["species"])
        self.T: float = phase["state"]["T"]
        self.P: float = phase["state"]["P"]

        # Alternative: extract from cantera
        # import cantera as ct
        # gas = ct.Solution(chem)
        # self.n_species = gas.n_species

        # Load Dataset
        self.data: np.ndarray = np.load(data_path)
        # DIMENSION: n * ((T, P, Y[ns], H[ns])_in, (T, P, Y[ns], H[ns])_gt)
        assert self.data.shape[1] == 2 * (
            2 + 2 * self.n_species
        ), "n_species in dataset does not match config file"
        self.time_step = 1e-7  # TODO: load from dataset
        self.formation_enthalpies = np.load(formation_enthalpies_path)
        assert (
            self.formation_enthalpies.shape[0] == self.n_species
        ), "n_species in dataset does not match formation_enthalpies"
        # self.dims = (1, 2 * (2 + 2 * self.n_species))

        # Apply normalization to Y_in and Y_gt
        self.Y_in = self.data[:, 2 : 2 + self.n_species]
        self.Y_gt = self.data[:, 4 + 2 * self.n_species : 4 + 3 * self.n_species]
        # The mass fraction calculated by cantera is not guaranteed to be positive.
        # This affects the boxcox transformation since it requires positive input.
        self.Y_gt[self.Y_gt < 0] = 0  # Or box-cox transformation would fail

        # Notation: Y -- boxcox --> Y_t -- norm --> Y_norm, Y_t_mean, Y_t_std
        self.lmbda = lmbda
        self.transform = lambda x: boxcox(x, lmbda)
        self.inv_transform = lambda y: inv_boxcox(y, lmbda)
        self.norm = (
            lambda x, mean, std: (x - mean) / std
        )  # TODO: handle the case for std=0
        self.denorm = lambda y, mean, std: y * std + mean

        Y_in_t = boxcox(self.Y_in, lmbda)
        self.Y_in_t_mean = Y_in_t.mean(axis=0)
        self.Y_in_t_std = Y_in_t.std(axis=0, ddof=1)
        Y_in_norm = self.norm(
            self.transform(self.Y_in), self.Y_in_t_mean, self.Y_in_t_std
        )
        Y_gt_t = boxcox(self.Y_gt, lmbda)
        self.Y_gt_t_mean = Y_gt_t.mean(axis=0)
        self.Y_gt_t_std = Y_gt_t.std(axis=0, ddof=1)
        Y_gt_norm = self.norm(
            self.transform(self.Y_gt), self.Y_gt_t_mean, self.Y_gt_t_std
        )

        Y_t_delta = Y_gt_t - Y_in_t
        self.Y_t_delta_mean = Y_t_delta.mean(axis=0)
        self.Y_t_delta_std = Y_t_delta.std(axis=0, ddof=1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index: int):
        frame = torch.tensor(self.data[index], dtype=torch.get_default_dtype())
        return frame.split(
            [
                1, # T_in
                1, # P_in
                self.n_species, # Y_in[ns]
                self.n_species, # H_in[ns]
                1, # T_gt
                1, # P_gt
                self.n_species, # Y_gt[ns]
                self.n_species, # H_gt[ns]
            ],
        )


class DFDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path=Path("dataset.npy"),
        config_path=Path("HyChem41s.yaml"),
        formation_enthalpies_path=Path("formation_enthalpies.npy"),
        lmbda=0.05,
        batch_size=32,
        num_workers=4,
    ):
        super().__init__()
        self.dataset = DFDataSet(
            data_path, config_path, formation_enthalpies_path, lmbda
        )
        self.train_set, self.val_set, self.test_set = random_split(
            self.dataset, [0.7, 0.2, 0.1]
        )
        self.dataloader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": True,
        }
        self.save_hyperparameters()

    # def setup(self, stage=None):
    #   pass

    def train_dataloader(self):
        return DataLoader(self.train_set, **self.dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_set, **self.dataloader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_set, **self.dataloader_kwargs)

    # TODO: predict_dataloader


def cli_main():
    # CSVLogger supports auto versioning;
    # WandbLogger does not, but stores hyperparameters in its own config file.
    cli = LightningCLI(DFNN, DFDataModule, save_config_callback=None)


if __name__ == "__main__":
    cli_main()
