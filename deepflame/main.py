from lightning.pytorch.cli import LightningCLI
from deepflame.model import DFNN
from deepflame.data import DFDataModule


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # 'data_config' is used for maintaining the consistency between the data and the model; it is a no-op.
        parser.add_argument("--data_config")


def cli_main():
    # CSVLogger supports auto versioning;
    # WandbLogger does not, but stores hyperparameters in its own config file.
    cli = MyLightningCLI(
        DFNN,
        DFDataModule,
        save_config_callback=None,
    )


# TODO - multiple model support: https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_intermediate_2.html#classes-from-any-package

if __name__ == "__main__":
    cli_main()
