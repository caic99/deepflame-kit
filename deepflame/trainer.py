import torch
from torch import nn
import lightning as L

from deepflame.data import DFDataSet
from deepflame.utils import normalize, denormalize, boxcox, inv_boxcox


class Trainer(L.LightningModule):
    dataset: DFDataSet
    model: torch.nn.Module
    trainer: L.Trainer

    def __init__(self):
        super().__init__()

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
    #     lr_scheduler = torch.optim.lr_scheduler.StepLR(
    #         optimizer, step_size=1, gamma=0.8
    #     )
    #     return [optimizer], [lr_scheduler]

    def setup(self, stage):
        # Transfer metadata to device
        self.dataset = self.trainer.datamodule.dataset  # type: ignore
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

        Y_gt_t = boxcox(Y_gt)
        loss1 = criterion(Y_pred_t, Y_gt_t)

        Y_pred = inv_boxcox(Y_pred_t)
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
