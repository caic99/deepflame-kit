from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import lightning as L
import yaml
from typing import Dict

from deepflame.utils import boxcox, inv_boxcox, normalize


class DFDataSet(Dataset):
    data:torch.Tensor
    stats:Dict[str, torch.Tensor]

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
        # self.T: float = phase["state"]["T"]
        # self.P: float = phase["state"]["P"]

        # Alternative: extract from cantera
        # import cantera as ct
        # gas = ct.Solution(chem)
        # self.n_species = gas.n_species

        # Load Dataset
        self.data = torch.tensor(np.load(data_path))
        # DIMENSION: n * ((T, P, Y[ns], H[ns])_in, (T, P, Y[ns], H[ns])_out)
        self.time_step = 1e-7  # TODO: load from dataset
        self.formation_enthalpies = torch.tensor(np.load(formation_enthalpies_path))
        assert (
            self.formation_enthalpies.shape[0] == self.n_species
        ), "n_species in dataset does not match formation_enthalpies"
        # self.dims = (1, 2 * (2 + 2 * self.n_species))
        assert (
            self.data.shape[1] == 4 + 4 * self.n_species
        ), "n_species in dataset does not match config file"
        self.indices = (
            1,  # T_in
            1,  # P_in
            self.n_species,  # Y_in[ns]
            self.n_species,  # H_in[ns]
            1,  # T_out
            1,  # P_out
            self.n_species,  # Y_label[ns]
            self.n_species,  # H_out[ns]
        )
        (T_in, P_in, Y_in, H_in, T_label, P_label, Y_label, H_label) = self.data.split(
            self.indices, dim=1
        )
        # The mass fraction calculated by cantera is not guaranteed to be positive.
        # This affects the boxcox transformation since it requires positive input.
        Y_label[Y_label < 0] = 0

        # Notation: Y -- boxcox --> Y_t -- norm --> Y_n, Y_t_mean, Y_t_std
        # Y_dt := (Y_t_out - Y_t_in) -- norm -> Y_dt_n, Y_dt_mean, Y_dt_std

        # The statistics marked as self.stats.*:dict[str, np.ndarray] loads into model buffer on training `setup()`.
        self.stats = {}

        def set_stats(self, key: str, value=None):
            if value is None:
                value = getattr(self, key)
            self.stats[key] = value

        def set_norm_stats(self, key: str, value=None):
            if value is None:
                value = getattr(self, key)
            assert value is not None
            set_stats(self, f"{key}_mean", value.mean(dim=0))
            set_stats(self, f"{key}_std", value.std(dim=0))

        set_stats(self, "lmbda", lmbda)
        set_stats(self, "formation_enthalpies")
        set_stats(self, "time_step")
        set_norm_stats(self, "T_in", T_in)
        set_norm_stats(self, "P_in", P_in)
        self.Y_t_in = boxcox(Y_in, lmbda)
        set_norm_stats(self, "Y_t_in")
        self.Y_t_out = boxcox(Y_label, lmbda)
        self.Y_dt = self.Y_t_out - self.Y_t_in
        set_norm_stats(self, "Y_dt")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index: int):
        frame = self.data[index].detach().clone().type(torch.get_default_dtype())
        return frame.split(self.indices)


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

    # def predict_dataloader(self):
    #     # TODO: predict_dataloader
    #     return DataLoader(self.test_set, **self.dataloader_kwargs)
    #     # return a mocked dataloader
