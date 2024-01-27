import torch
from torch import nn
import lightning as L

from model import MLP
from data import DFDataSet

class DFNN(L.LightningModule):
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
