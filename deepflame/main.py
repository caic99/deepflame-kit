from lightning.pytorch.cli import LightningCLI
from deepflame.model import DFNN
from deepflame.data import DFDataModule


def cli_main():
    # CSVLogger supports auto versioning;
    # WandbLogger does not, but stores hyperparameters in its own config file.
    cli = LightningCLI(DFNN, DFDataModule, save_config_callback=None)


if __name__ == "__main__":
    cli_main()
